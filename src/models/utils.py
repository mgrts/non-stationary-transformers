import json
import logging
import os

import mlflow
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import LEAVE_RATIO, NUM_VIS_EXAMPLES, PATIENCE
from src.models.model import TransformerWithPE
from src.visualization.visualize import visualize_prediction

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def load_dataset(path):
    """Load a TensorDataset saved by make_dataset.py.

    PyTorch >= 2.6 defaults ``torch.load`` to ``weights_only=True``, which
    refuses to unpickle the ``TensorDataset`` objects we save. These files are
    produced by our own pipeline (trusted), so we load them with
    ``weights_only=False``.
    """
    return torch.load(path, weights_only=False)


def load_data_meta(path):
    """Read the data-generation condition sidecar written by the data pipeline.

    Lets trainers log the TRUE condition the data was built with (alphas,
    stability_period, smoothing_type, data_type) instead of re-reading their own
    CLI flags, which can silently drift from how the data was generated. Returns
    an empty dict if the sidecar is missing (older datasets / partial runs).
    """
    if not os.path.exists(path):
        logger.warning(
            "Data metadata sidecar %s not found; logged condition params may be incomplete. "
            "Re-run generate_data.py + process_data.py to produce it.",
            path,
        )
        return {}
    with open(path) as f:
        return json.load(f)


# --- Sequence splitting -------------------------------------------------------
def split_sequence_with_decoder(sequence, leave_ratio=LEAVE_RATIO):
    """3-way split for teacher forcing: `src`, decoder input `tgt`, target `tgt_y`.

    The split point is derived from the actual sequence length (no hardcoded
    length). `tgt` is `tgt_y` shifted right by one - it starts with the last
    token of `src` and ends with the second-last token of `tgt_y`.
    """
    split_index = int(sequence.shape[1] * leave_ratio)
    src = sequence[:, :split_index, :]
    tgt = sequence[:, split_index - 1 : -1, :]
    tgt_y = sequence[:, split_index:, :]
    return src, tgt, tgt_y


def move_to_device(device: torch.device, *tensors: torch.Tensor) -> list[torch.Tensor]:
    moved_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            moved_tensors.append(tensor.to(device))
        else:
            moved_tensors.append(tensor)
    return moved_tensors


def prepare_batch(sequence, device, leave_ratio=LEAVE_RATIO):
    """3-way teacher-forcing split + device move, shared by both architectures."""
    src, tgt, tgt_y = split_sequence_with_decoder(sequence, leave_ratio)
    src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)
    return src, tgt, tgt_y


def model_forward(model, src, tgt, tgt_y):
    """Teacher-forced forward pass for either architecture (length = tgt_y)."""
    if isinstance(model, TransformerWithPE):
        return model(src, tgt)
    return model(src, output_sequence_length=tgt_y.shape[1], tgt=tgt)


# --- Losses & metrics ---------------------------------------------------------
def mae_loss(output, target):
    return torch.mean(torch.abs(output - target))


def rmse_loss(output, target):
    return torch.sqrt(torch.mean((output - target) ** 2))


def mape_loss(output, target, eps=1e-2):
    # On normalized (~0-centered) targets, dividing by values near zero makes
    # MAPE explode, so near-zero targets are masked out. CAVEAT: the surviving
    # subset is data-dependent, so MAPE summarizes a different set of points than
    # the full-set MSE/MAE/RMSE/SMAPE and is not directly comparable to them.
    # It is reported only as a rough secondary indicator - prefer MSE/MAE/RMSE
    # (and SMAPE) on this normalized data. NaN if every target is masked.
    mask = target.abs() > eps
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=target.device)
    return torch.mean(torch.abs((target[mask] - output[mask]) / target[mask])) * 100


def smape_loss(output, target):
    return (
        torch.mean(
            2 * torch.abs(target - output) / (torch.abs(target) + torch.abs(output) + 1e-10)
        )
        * 100
    )


class CauchyLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction="mean"):
        super(CauchyLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        diffs = input - target
        cauchy_losses = self.gamma * torch.log(1 + (diffs**2) / self.gamma)
        if self.reduction == "sum":
            return cauchy_losses.sum()
        elif self.reduction == "mean":
            return cauchy_losses.mean()
        else:
            return cauchy_losses


def make_criterion(loss_type, cauchy_gamma=1.0):
    """Single source of truth for the training criterion, shared by both trainers."""
    if loss_type == "MSE":
        return nn.MSELoss()
    elif loss_type == "L1":
        return nn.L1Loss()
    elif loss_type == "Cauchy":
        return CauchyLoss(gamma=cauchy_gamma)
    raise ValueError(f"Loss type {loss_type} is not supported.")


def error_metrics(pred, target):
    """Forecast error metrics, computed on whatever scale `pred`/`target` are in.

    In this project that is the causally-normalized (and log1p-for-counts) scale
    used during training - the fitted per-sequence scalers are not retained, so
    these are NORMALIZED-scale errors, not original units. MSE/MAE/RMSE are
    therefore scale-dependent (comparable across models on the same data, not
    across differently-scaled datasets); MAPE/SMAPE are scale-free but weak on
    ~0-centered data (see mape_loss).
    """
    return {
        "mse": torch.mean((pred - target) ** 2).item(),
        "mae": mae_loss(pred, target).item(),
        "rmse": rmse_loss(pred, target).item(),
        "mape": mape_loss(pred, target).item(),
        "smape": smape_loss(pred, target).item(),
    }


# --- Training & evaluation ----------------------------------------------------
def _validate(model, criterion, loader, device):
    """Average AUTOREGRESSIVE validation loss - the honest forecasting objective
    used for early stopping and model selection (not the optimistic
    teacher-forced loss)."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            src, _, tgt_y = prepare_batch(batch[0], device)
            pred = model.infer(src, tgt_y.shape[1])
            total += criterion(pred, tgt_y).item()
            n += 1
    return total / max(n, 1)


def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    split_name,
    num_epoch,
    device,
    val_loader=None,
    patience=PATIENCE,
):
    """Train with optional validation-based early stopping.

    When `val_loader` is given, the model is selected by the best autoregressive
    validation loss and restored to that checkpoint at the end; training stops
    early after `patience` epochs without improvement.
    """
    n_batches = len(train_loader)
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0.0
        with tqdm(
            total=n_batches, desc=f"Epoch {epoch + 1}/{num_epoch} for {split_name}", unit="batch"
        ) as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                src, tgt, tgt_y = prepare_batch(batch[0], device)
                pred = model_forward(model, src, tgt, tgt_y)
                loss = criterion(pred, tgt_y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        avg_epoch_loss = epoch_loss / n_batches
        mlflow.log_metric(f"train_loss_{split_name}", avg_epoch_loss, step=epoch)

        if val_loader is not None:
            val_loss = _validate(model, criterion, val_loader, device)
            mlflow.log_metric(f"val_loss_{split_name}", val_loss, step=epoch)
            logger.info(
                f"[{split_name}] Epoch {epoch + 1}/{num_epoch} - "
                f"train_loss: {avg_epoch_loss:.4f}, val_loss (autoregressive): {val_loss:.4f}"
            )

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"[{split_name}] Early stopping at epoch {epoch + 1} "
                        f"(no val improvement for {patience} epochs)."
                    )
                    break
        else:
            logger.info(
                f"[{split_name}] Epoch {epoch + 1}/{num_epoch} - train_loss: {avg_epoch_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        mlflow.log_metric(f"best_val_loss_{split_name}", best_val)
        logger.info(f"[{split_name}] Restored best checkpoint (val_loss={best_val:.4f}).")

    return best_val if best_state is not None else avg_epoch_loss


def evaluate_model(model, criterion, loader, split_name, device, vis_prefix=None):
    """Evaluate on a held-out set, reporting BOTH protocols.

    The AUTOREGRESSIVE ("infer") metrics are the primary result - they reflect
    true multi-step forecasting where the model consumes its own predictions.
    The teacher-forced ("tf") metrics are reported alongside as an optimistic
    upper bound (the model sees ground-truth decoder inputs) and should not be
    quoted as the forecasting performance.
    """
    n_batches = len(loader)
    model.eval()

    # Accumulate sample-WEIGHTED metric sums (weight = batch size) so the final
    # average is a true per-sample mean, not a per-batch mean that over-weights a
    # smaller trailing batch. RMSE is derived from the aggregated MSE at the end
    # (sqrt of a weighted MSE mean; averaging per-batch RMSEs would be wrong).
    tf_sums = {"mse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0}
    ar_sums = {"mse": 0.0, "mae": 0.0, "mape": 0.0, "smape": 0.0}
    total = 0

    with torch.no_grad():
        with tqdm(total=n_batches, desc=f"Evaluating {split_name} dataset", unit="batch") as pbar:
            for idx, batch in enumerate(loader):
                src, tgt, tgt_y = prepare_batch(batch[0], device)
                bs = tgt_y.shape[0]
                total += bs

                pred_tf = model_forward(model, src, tgt, tgt_y)
                for k, v in error_metrics(pred_tf, tgt_y).items():
                    if k in tf_sums:
                        tf_sums[k] += v * bs

                pred_infer = model.infer(src, tgt_y.shape[1])
                for k, v in error_metrics(pred_infer, tgt_y).items():
                    if k in ar_sums:
                        ar_sums[k] += v * bs

                if idx < NUM_VIS_EXAMPLES:
                    figure = visualize_prediction(src, tgt_y, pred_tf, pred_infer)
                    name = f"prediction_{vis_prefix or split_name}_{idx}.png"
                    mlflow.log_figure(figure, name)

                pbar.update(1)

    denom = max(total, 1)
    tf = {k: s / denom for k, s in tf_sums.items()}
    ar = {k: s / denom for k, s in ar_sums.items()}
    tf["rmse"] = tf["mse"] ** 0.5
    ar["rmse"] = ar["mse"] ** 0.5

    logger.info(
        f"{split_name.capitalize()} AUTOREGRESSIVE (primary) - "
        f"MSE: {ar['mse']:.4f}, MAE: {ar['mae']:.4f}, RMSE: {ar['rmse']:.4f}, "
        f"MAPE: {ar['mape']:.2f}, SMAPE: {ar['smape']:.2f}"
    )
    logger.info(
        f"{split_name.capitalize()} teacher-forced (optimistic) - "
        f"MSE: {tf['mse']:.4f}, MAE: {tf['mae']:.4f}, RMSE: {tf['rmse']:.4f}, "
        f"MAPE: {tf['mape']:.2f}, SMAPE: {tf['smape']:.2f}"
    )

    metrics = {}
    for k, v in ar.items():
        metrics[f"{split_name}_ar_{k}"] = v
    for k, v in tf.items():
        metrics[f"{split_name}_tf_{k}"] = v
    mlflow.log_metrics(metrics)
    return metrics
