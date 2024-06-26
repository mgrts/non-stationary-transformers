import logging

import click
import mlflow
import numpy as np
import torch
import torch.nn as nn
from model import TransformerWithPE
from torch.utils.data import DataLoader

from src.config import (BS, CAUCHY_LOSS_GAMMA, DATA_TYPE, FEATURE_DIM,
                        FINAL_ALPHA, INITIAL_ALPHA, KERNEL_SIZE, LOSS_TYPE, LR,
                        N_TIME_SERIES, NUM_EPOCHS, NUM_FEATURES, NUM_HEADS,
                        NUM_LAYERS, NUM_VIS_EXAMPLES, RANDOM_STATE,
                        SEQUENCE_LENGTH, SMOOTHING_TYPE, STABILITY_PERIOD,
                        TEST_DATA_PATH, TRACKING_URI, TRAIN_DATA_PATH)
from src.visualization.visualize import visualize_prediction

EXPERIMENT_NAME = 'initial experiments'


def mape_loss(output, target):
    """
    Calculate the Mean Absolute Percentage Error between output and target
    """
    return torch.mean(torch.abs((target - output) / target)) * 100


def smape_loss(output, target):
    """
    Calculate the Symmetric Mean Absolute Percentage Error between output and target
    """
    return torch.mean(2 * torch.abs(target - output) / (torch.abs(target) + torch.abs(output))) * 100


def move_to_device(device: torch.device, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """Move all given tensors to the given device.

    Args:
        device: device to move tensors to
        tensors: tensors to move

    Returns:
        list[torch.Tensor]: moved tensors
    """
    moved_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            moved_tensors.append(tensor.to(device))
        else:
            moved_tensors.append(tensor)
    return moved_tensors


def split_sequence(
        sequence: np.ndarray, ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a sequence into 2 (3) parts, as is required by our transformer
    model.

    Assume our sequence length is L, we then split this into src of length N
    and tgt_y of length M, with N + M = L.
    src, the first part of the input sequence, is the input to the encoder, and we
    expect the decoder to predict tgt_y, the second part of the input sequence.
    In addition, we generate tgt, which is tgt_y but "shifted left" by one - i.e. it
    starts with the last token of src, and ends with the second-last token in tgt_y.
    This sequence will be the input to the decoder.


    Args:
        sequence: batched input sequences to split [bs, seq_len, num_features]
        ratio: split ratio, N = ratio * L

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: src, tgt, tgt_y
    """
    src_end = int(sequence.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    src = sequence[:, :src_end]
    # [bs, tgt_seq_len, num_features]
    tgt = sequence[:, src_end - 1: -1]
    # [bs, tgt_seq_len, num_features]
    tgt_y = sequence[:, src_end:]

    return src, tgt, tgt_y


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


@click.command()
@click.option('--smoothing-type', default=SMOOTHING_TYPE,
              help='Type of smoothing to apply (gaussian, sine, cosine, combined_cosine_gaussian)')
@click.option('--stability-period', default=STABILITY_PERIOD,
              type=click.Choice(['short', 'moderate', 'long'], case_sensitive=False),
              help='Period of stability (short, moderate, long)')
@click.option('--initial-alpha', default=INITIAL_ALPHA, type=float, help='Initial alpha value for generating sequences')
@click.option('--final-alpha', default=FINAL_ALPHA, type=float, help='Final alpha value for generating sequences')
def main(smoothing_type, stability_period, initial_alpha, final_alpha):
    logger = logging.getLogger(__name__)

    logger.info('Training model')

    train_set = torch.load(TRAIN_DATA_PATH)
    test_set = torch.load(TEST_DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        run_id = run.info.run_id
        # artifacts_dir = os.path.join(TRACKING_URI, run_id)

        logger.info(f'MLflow Run ID: {run_id}')

        # Log parameters
        mlflow.log_params({
            'learning_rate': LR,
            'batch_size': BS,
            'num_epochs': NUM_EPOCHS,
            'loss_type': LOSS_TYPE,
            'random_state': RANDOM_STATE,
            'data_type': DATA_TYPE,
            'smoothing_type': smoothing_type,
            'kernel_size': KERNEL_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
            'n_time_series': N_TIME_SERIES,
            'stability_period': stability_period,
            'initial_alpha': initial_alpha,
            'final_alpha': final_alpha,
        })

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model, optimizer and loss criterion
        model = TransformerWithPE(
            NUM_FEATURES, NUM_FEATURES, FEATURE_DIM, NUM_HEADS, NUM_LAYERS
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        if LOSS_TYPE == 'MSE':
            criterion = torch.nn.MSELoss()
        elif LOSS_TYPE == 'L1':
            criterion = torch.nn.L1Loss()
        elif LOSS_TYPE == 'Cauchy':
            criterion = CauchyLoss(gamma=CAUCHY_LOSS_GAMMA)
            mlflow.log_param('cauchy_loss_gamma', CAUCHY_LOSS_GAMMA)
        else:
            raise AttributeError(f'Loss type {LOSS_TYPE} is not supported.')

        # Train loop
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            epoch_mape = 0.0
            epoch_smape = 0.0
            for batch in train_loader:
                optimizer.zero_grad()

                src, tgt, tgt_y = split_sequence(batch[0])
                src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)
                # [bs, tgt_seq_len, num_features]
                pred = model(src, tgt)
                loss = criterion(pred, tgt_y)
                epoch_loss += loss.item()

                mape = mape_loss(pred, tgt_y)
                smape = smape_loss(pred, tgt_y)
                epoch_mape += mape.item()
                epoch_smape += smape.item()

                loss.backward()
                optimizer.step()

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_mape = epoch_mape / len(train_loader)
            avg_epoch_smape = epoch_smape / len(train_loader)

            logger.info(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}')
            logger.info(f'MAPE: {avg_epoch_mape: .2f}, SMAPE: {avg_epoch_smape: .2f}')

            mlflow.log_metric('loss', avg_epoch_loss, step=epoch)
            mlflow.log_metric('mape', avg_epoch_mape, step=epoch)
            mlflow.log_metric('smape', avg_epoch_smape, step=epoch)

        # Evaluate model
        model.eval()
        eval_loss = 0.0
        eval_mape = 0.0
        eval_smape = 0.0
        infer_loss = 0.0
        infer_mape = 0.0
        infer_smape = 0.0

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                src, tgt, tgt_y = split_sequence(batch[0])
                src, tgt, tgt_y = move_to_device(device, src, tgt, tgt_y)

                # [bs, tgt_seq_len, num_features]
                pred = model(src, tgt)
                loss = criterion(pred, tgt_y)
                eval_loss += loss.item()
                eval_mape += mape_loss(pred, tgt_y).item()
                eval_smape += smape_loss(pred, tgt_y).item()

                # Run inference with model
                pred_infer = model.infer(src, tgt.shape[1])
                loss_infer = criterion(pred_infer, tgt_y)
                infer_loss += loss_infer.item()
                infer_mape += mape_loss(pred_infer, tgt_y).item()
                infer_smape += smape_loss(pred_infer, tgt_y).item()

                if idx < NUM_VIS_EXAMPLES:
                    figure = visualize_prediction(src, tgt, pred, pred_infer)
                    mlflow.log_figure(figure, f'prediction_{idx}.png')

        avg_eval_loss = eval_loss / len(test_loader)
        avg_eval_mape = eval_mape / len(test_loader)
        avg_eval_smape = eval_smape / len(test_loader)
        avg_infer_loss = infer_loss / len(test_loader)
        avg_infer_mape = infer_mape / len(test_loader)
        avg_infer_smape = infer_smape / len(test_loader)

        logger.info(
            f'Evaluation Metrics - Loss: {avg_eval_loss:.4f}, MAPE: {avg_eval_mape:.2f}, SMAPE: {avg_eval_smape:.2f}')
        logger.info(
            f'Inference Metrics - Loss: {avg_infer_loss:.4f}, MAPE: {avg_infer_mape:.2f}, SMAPE: {avg_infer_smape:.2f}')

        mlflow.log_metrics({
            'eval_loss': avg_eval_loss,
            'eval_mape': avg_eval_mape,
            'eval_smape': avg_eval_smape,
            'infer_loss': avg_infer_loss,
            'infer_mape': avg_infer_mape,
            'infer_smape': avg_infer_smape
        })

        # Log the model
        mlflow.pytorch.log_model(model, 'model')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
