import logging

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import NUM_VIS_EXAMPLES
from src.visualization.visualize import visualize_prediction

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def split_sequence(sequence, sequence_length=300, leave_ratio=0.8):
    split_index = int(sequence_length * leave_ratio)
    src = sequence[:, :split_index, :]
    tgt_y = sequence[:, split_index:, :]
    return src, tgt_y


def move_to_device(device: torch.device, *tensors: torch.Tensor) -> list[torch.Tensor]:
    moved_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            moved_tensors.append(tensor.to(device))
        else:
            moved_tensors.append(tensor)
    return moved_tensors


def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / (target + 1e-10))) * 100


def smape_loss(output, target):
    return torch.mean(2 * torch.abs(target - output) / (torch.abs(target) + torch.abs(output))) * 100


class CauchyLoss(nn.Module):
    def __init__(self, gamma=1.0, reduction='mean'):
        super(CauchyLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        diffs = input - target
        cauchy_losses = self.gamma * torch.log(1 + (diffs ** 2) / self.gamma)
        if self.reduction == 'sum':
            return cauchy_losses.sum()
        elif self.reduction == 'mean':
            return cauchy_losses.mean()
        else:
            return cauchy_losses


def pad_or_truncate(sequence, target_length):
    if len(sequence) > target_length:
        return sequence[:target_length]
    elif len(sequence) < target_length:
        return np.pad(sequence, ((0, target_length - len(sequence)), (0, 0)), mode='constant')
    return sequence


def normalize_time_series(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        num_samples, seq_len = data.shape
        normalized_data = np.zeros_like(data)
        for i in range(num_samples):
            scaler = StandardScaler()
            normalized_data[i, :] = scaler.fit_transform(data[i, :].reshape(-1, 1)).flatten()
    elif data.ndim == 3:
        num_samples, seq_len, num_features = data.shape
        normalized_data = np.zeros_like(data)
        for i in range(num_samples):
            scaler = StandardScaler()
            normalized_data[i, :, :] = scaler.fit_transform(data[i, :, :])
    else:
        raise ValueError("Data must be either 2 or 3 dimensions")

    return normalized_data


def normalize_data(data: np.ndarray, sequence_length: int = 300) -> np.ndarray:
    data = normalize_time_series(data)
    if data.ndim == 2:
        data = np.array([pad_or_truncate(seq, sequence_length) for seq in data])
    elif data.ndim == 3:
        data = np.array([pad_or_truncate(seq, sequence_length) for seq in data])

    if data.ndim == 2:
        data = data[..., np.newaxis]
    return data


def train_model(model, optimizer, criterion, loader, split_name, num_epoch, device):
    n_batches = len(loader)

    model.train()
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        epoch_mape = 0.0
        epoch_smape = 0.0
        with tqdm(total=n_batches, desc=f"Epoch {epoch + 1}/{num_epoch} for {split_name}",
                  unit='batch') as pbar:
            for batch in loader:
                optimizer.zero_grad()
                src, tgt_y = split_sequence(batch[0])
                src, tgt_y = move_to_device(device, src, tgt_y)
                pred = model(src)
                loss = criterion(pred, tgt_y)
                epoch_loss += loss.item()
                mape = mape_loss(pred, tgt_y)
                smape = smape_loss(pred, tgt_y)
                epoch_mape += mape.item()
                epoch_smape += smape.item()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        avg_epoch_loss = epoch_loss / n_batches
        avg_epoch_mape = epoch_mape / n_batches
        avg_epoch_smape = epoch_smape / n_batches
        logger.info(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {avg_epoch_loss:.4f}')
        logger.info(f'MAPE: {avg_epoch_mape:.2f}, SMAPE: {avg_epoch_smape:.2f}')
        mlflow.log_metric(f'loss_{split_name}', avg_epoch_loss, step=epoch)
        mlflow.log_metric(f'mape_{split_name}', avg_epoch_mape, step=epoch)
        mlflow.log_metric(f'smape_{split_name}', avg_epoch_smape, step=epoch)


def evaluate_model(model, criterion, loader, split_name, device):
    n_batches = len(loader)

    model.eval()
    eval_loss = 0.0
    eval_mape = 0.0
    eval_smape = 0.0
    infer_loss = 0.0
    infer_mape = 0.0
    infer_smape = 0.0

    with torch.no_grad():
        with tqdm(total=n_batches, desc=f"Evaluating {split_name} dataset", unit='sample') as pbar:
            for idx, batch in enumerate(loader):
                src, tgt_y = split_sequence(batch[0])
                src, tgt_y = move_to_device(device, src, tgt_y)

                pred = model(src)
                loss = criterion(pred, tgt_y)
                eval_loss += loss.item()
                eval_mape += mape_loss(pred, tgt_y).item()
                eval_smape += smape_loss(pred, tgt_y).item()

                pred_infer = model.infer(src, tgt_y.shape[1])
                loss_infer = criterion(pred_infer, tgt_y)
                infer_loss += loss_infer.item()
                infer_mape += mape_loss(pred_infer, tgt_y).item()
                infer_smape += smape_loss(pred_infer, tgt_y).item()

                if idx < NUM_VIS_EXAMPLES:
                    figure = visualize_prediction(src, tgt_y, pred, pred_infer)
                    mlflow.log_figure(figure, f'prediction_{split_name}_{idx}.png')

                pbar.update(1)

    avg_eval_loss = eval_loss / n_batches
    avg_eval_mape = eval_mape / n_batches
    avg_eval_smape = eval_smape / n_batches
    avg_infer_loss = infer_loss / n_batches
    avg_infer_mape = infer_mape / n_batches
    avg_infer_smape = infer_smape / n_batches

    logging.info(
        f'{split_name.capitalize()} Evaluation Metrics - Loss: {avg_eval_loss:.4f}'
        f'MAPE: {avg_eval_mape:.2f}, SMAPE: {avg_eval_smape:.2f}'
    )
    logging.info(
        f'{split_name.capitalize()} Inference Metrics - Loss: {avg_infer_loss:.4f}'
        f'MAPE: {avg_infer_mape:.2f}, SMAPE: {avg_infer_smape:.2f}'
    )
    mlflow.log_metrics({
        f'{split_name}_eval_loss': avg_eval_loss,
        f'{split_name}_eval_mape': avg_eval_mape,
        f'{split_name}_eval_smape': avg_eval_smape,
        f'{split_name}_infer_loss': avg_infer_loss,
        f'{split_name}_infer_mape': avg_infer_mape,
        f'{split_name}_infer_smape': avg_infer_smape,
    })
