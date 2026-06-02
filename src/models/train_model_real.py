import logging

import click
import mlflow
import torch
from torch.utils.data import DataLoader

from src.config import (
    BS,
    CAUCHY_LOSS_GAMMA,
    COVID_SEQ_CHUNK_SIZE,
    DATA_TYPE,
    FEATURE_DIM,
    FINAL_ALPHA,
    INITIAL_ALPHA,
    KERNEL_SIZE,
    LEAVE_RATIO,
    LOSS_TYPE,
    LR,
    N_TIME_SERIES,
    NUM_FEATURES,
    NUM_HEADS,
    NUM_LAYERS,
    OWID_TEST_DATA_PATH,
    OWID_TRAIN_DATA_PATH,
    OWID_VAL_DATA_PATH,
    PATIENCE,
    REAL_BS,
    SEQUENCE_LENGTH,
    SMOOTHING_TYPE,
    STABILITY_PERIOD,
    SYNTHETIC_COVID_TEST_DATA_PATH,
    SYNTHETIC_COVID_TRAIN_DATA_PATH,
    SYNTHETIC_COVID_VAL_DATA_PATH,
    TRACKING_URI,
)
from src.models.model import LSTM, TransformerWithPE
from src.models.utils import evaluate_model, load_dataset, make_criterion, train_model
from src.seeding import seed_everything

EXPERIMENT_NAME = "transfer-covid"
PRE_TRAIN_NUM_EPOCHS = 3
FINE_TUNE_NUM_EPOCHS = 20
PRE_TRAIN_FIRST = True
MODEL_TYPE = "LSTM"


def build_model(model_type, device):
    if model_type == "LSTM":
        return LSTM(NUM_FEATURES, FEATURE_DIM, NUM_LAYERS, NUM_FEATURES).to(device)
    elif model_type == "Transformer":
        return TransformerWithPE(
            NUM_FEATURES, NUM_FEATURES, FEATURE_DIM, NUM_HEADS, NUM_LAYERS
        ).to(device)
    raise ValueError(f"Model type {model_type} is not supported.")


@click.command()
@click.option(
    "--smoothing-type",
    default=SMOOTHING_TYPE,
    help="Type of smoothing to apply (gaussian, sine, cosine, combined_cosine_gaussian)",
)
@click.option(
    "--stability-period",
    default=STABILITY_PERIOD,
    type=click.Choice(["short", "moderate", "long"], case_sensitive=False),
    help="Period of stability (short, moderate, long)",
)
@click.option(
    "--initial-alpha",
    default=INITIAL_ALPHA,
    type=float,
    help="Initial alpha value for generating sequences",
)
@click.option(
    "--final-alpha",
    default=FINAL_ALPHA,
    type=float,
    help="Final alpha value for generating sequences",
)
@click.option(
    "--model-type",
    default=MODEL_TYPE,
    type=click.Choice(["LSTM", "Transformer"], case_sensitive=True),
    help="Architecture to train",
)
@click.option(
    "--seed", default=None, type=int, help="Model-training seed (variance source across runs)"
)
def main(smoothing_type, stability_period, initial_alpha, final_alpha, model_type, seed):
    logger = logging.getLogger(__name__)
    logger.info("Starting the transfer-learning experiment")

    if seed is not None:
        seed_everything(seed)

    logger.info("Loading synthetic (pre-training) dataset")
    synthetic_train_set = load_dataset(SYNTHETIC_COVID_TRAIN_DATA_PATH)
    synthetic_val_set = load_dataset(SYNTHETIC_COVID_VAL_DATA_PATH)
    synthetic_test_set = load_dataset(SYNTHETIC_COVID_TEST_DATA_PATH)

    logger.info("Loading real-world (OWID) dataset")
    real_train_set = load_dataset(OWID_TRAIN_DATA_PATH)
    real_val_set = load_dataset(OWID_VAL_DATA_PATH)
    real_test_set = load_dataset(OWID_TEST_DATA_PATH)

    synthetic_train_loader = DataLoader(synthetic_train_set, batch_size=BS, shuffle=True)
    synthetic_val_loader = DataLoader(synthetic_val_set, batch_size=BS, shuffle=False)
    synthetic_test_loader = DataLoader(synthetic_test_set, batch_size=BS, shuffle=False)
    real_train_loader = DataLoader(real_train_set, batch_size=REAL_BS, shuffle=True)
    real_val_loader = DataLoader(real_val_set, batch_size=REAL_BS, shuffle=False)
    real_test_loader = DataLoader(real_test_set, batch_size=REAL_BS, shuffle=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(
            {
                "model_type": model_type,
                "learning_rate": LR,
                "loss_type": LOSS_TYPE,
                "seed": seed,
                "leave_ratio": LEAVE_RATIO,
                "n_time_series": N_TIME_SERIES,
                "covid_seq_chunk_size": COVID_SEQ_CHUNK_SIZE,
                "pre_train_first": PRE_TRAIN_FIRST,
                "pre_train_num_epochs": PRE_TRAIN_NUM_EPOCHS,
                "fine_tune_num_epochs": FINE_TUNE_NUM_EPOCHS,
                "patience": PATIENCE,
                "batch_size": BS,
                "real_batch_size": REAL_BS,
                "feature_dim": FEATURE_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "data_type": DATA_TYPE,
                "smoothing_type": smoothing_type,
                "kernel_size": KERNEL_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "stability_period": stability_period,
                "initial_alpha": initial_alpha,
                "final_alpha": final_alpha,
            }
        )
        if LOSS_TYPE == "Cauchy":
            mlflow.log_param("cauchy_loss_gamma", CAUCHY_LOSS_GAMMA)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(model_type, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = make_criterion(LOSS_TYPE, CAUCHY_LOSS_GAMMA)

        if PRE_TRAIN_FIRST:
            logger.info("Pre-training on synthetic dataset")
            train_model(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=synthetic_train_loader,
                val_loader=synthetic_val_loader,
                split_name="synthetic",
                num_epoch=PRE_TRAIN_NUM_EPOCHS,
                device=device,
            )

        logger.info("Fine-tuning on real-world dataset")
        train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=real_train_loader,
            val_loader=real_val_loader,
            split_name="real",
            num_epoch=FINE_TUNE_NUM_EPOCHS,
            device=device,
        )

        logger.info("Evaluating on synthetic test set")
        evaluate_model(model, criterion, synthetic_test_loader, "synthetic_test", device)

        logger.info("Evaluating on real-world test set")
        evaluate_model(model, criterion, real_test_loader, "real_test", device)

        logger.info("Saving model")
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
