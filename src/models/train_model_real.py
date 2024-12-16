import logging

import click
import mlflow
import torch
from torch.utils.data import DataLoader

from src.config import (BS, CAUCHY_LOSS_GAMMA, COVID_SEQ_CHUNK_SIZE, DATA_TYPE,
                        FEATURE_DIM, FINAL_ALPHA, INITIAL_ALPHA, KERNEL_SIZE,
                        LOSS_TYPE, LR, N_TIME_SERIES, NUM_FEATURES, NUM_HEADS,
                        NUM_LAYERS, OWID_TEST_DATA_PATH, OWID_TRAIN_DATA_PATH,
                        RANDOM_STATE, SEQUENCE_LENGTH, SMOOTHING_TYPE,
                        STABILITY_PERIOD, SYNTHETIC_COVID_TEST_DATA_PATH,
                        SYNTHETIC_COVID_TRAIN_DATA_PATH, TRACKING_URI)
from src.models.model import LSTM, TransformerWithPE
from src.models.utils import CauchyLoss, evaluate_model, train_model

EXPERIMENT_NAME = 'lstm'
PRE_TRAIN_NUM_EPOCHS = 3
FINE_TUNE_NUM_EPOCHS = 20
PRE_TRAIN_FIRST = True
MODEL_TYPE = 'LSTM'


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
    logger.info('Starting the main function')

    logger.info('Loading synthetic dataset')
    synthetic_train_set = torch.load(SYNTHETIC_COVID_TRAIN_DATA_PATH)
    synthetic_test_set = torch.load(SYNTHETIC_COVID_TEST_DATA_PATH)

    logger.info('Loading real-world dataset')
    real_train_set = torch.load(OWID_TRAIN_DATA_PATH)
    real_test_set = torch.load(OWID_TEST_DATA_PATH)

    synthetic_train_loader = DataLoader(synthetic_train_set, batch_size=BS, shuffle=True)
    synthetic_test_loader = DataLoader(synthetic_test_set, batch_size=BS, shuffle=False)
    real_train_loader = DataLoader(real_train_set, batch_size=16, shuffle=True)
    real_test_loader = DataLoader(real_test_set, batch_size=16, shuffle=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        run_id = run.info.run_id

        logger.info(f'MLflow Run ID: {run_id}')

        mlflow.log_params({
            'learning_rate': LR,
            'loss_type': LOSS_TYPE,
            'random_state': RANDOM_STATE,
            'n_time_series': N_TIME_SERIES,
            'covid_seq_chunk_size': COVID_SEQ_CHUNK_SIZE,
            'pre_train_first': PRE_TRAIN_FIRST,
            'batch_size': BS,
            'data_type': DATA_TYPE,
            'smoothing_type': smoothing_type,
            'kernel_size': KERNEL_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
            'stability_period': stability_period,
            'initial_alpha': initial_alpha,
            'final_alpha': final_alpha,
        })

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if MODEL_TYPE == 'LSTM':
            model = LSTM(NUM_FEATURES, FEATURE_DIM, NUM_LAYERS, NUM_FEATURES).to(device)
        elif MODEL_TYPE == 'Transformer':
            model = TransformerWithPE(NUM_FEATURES, NUM_FEATURES, FEATURE_DIM, NUM_HEADS, NUM_LAYERS).to(device)
        else:
            raise AttributeError(f'Model type {MODEL_TYPE} is not supported.')

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

        if PRE_TRAIN_FIRST:
            logger.info('Pre-training on synthetic dataset')
            train_model(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                loader=synthetic_train_loader,
                split_name='synthetic',
                num_epoch=PRE_TRAIN_NUM_EPOCHS,
                device=device
            )

        logger.info('Fine-tuning on real-world dataset')
        train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loader=real_train_loader,
            split_name='real',
            num_epoch=FINE_TUNE_NUM_EPOCHS,
            device=device
        )

        logger.info('Evaluating model on synthetic dataset')
        evaluate_model(
            model=model,
            criterion=criterion,
            loader=synthetic_test_loader,
            split_name='synthetic',
            device=device
        )

        logger.info('Evaluating model on real-world dataset')
        evaluate_model(
            model=model,
            criterion=criterion,
            loader=real_test_loader,
            split_name='real',
            device=device
        )

        logger.info('Saving model')
        mlflow.pytorch.log_model(model, 'model')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
