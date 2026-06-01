import logging

import click
import mlflow
import torch
from torch.utils.data import DataLoader

from src.config import (
    BS,
    CAUCHY_LOSS_GAMMA,
    DATA_TYPE,
    FEATURE_DIM,
    FINAL_ALPHA,
    INITIAL_ALPHA,
    KERNEL_SIZE,
    LEAVE_RATIO,
    LOSS_TYPE,
    LR,
    N_TIME_SERIES,
    NUM_EPOCHS,
    NUM_FEATURES,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    SEQUENCE_LENGTH,
    SMOOTHING_TYPE,
    STABILITY_PERIOD,
    TEST_DATA_PATH,
    TRACKING_URI,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from src.models.model import TransformerWithPE
from src.models.utils import evaluate_model, make_criterion, train_model
from src.seeding import seed_everything

EXPERIMENT_NAME = "initial experiments"


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
    "--seed", default=None, type=int, help="Model-training seed (variance source across runs)"
)
def main(smoothing_type, stability_period, initial_alpha, final_alpha, seed):
    logger = logging.getLogger(__name__)
    logger.info("Training model")

    if seed is not None:
        seed_everything(seed)

    train_set = torch.load(TRAIN_DATA_PATH)
    val_set = torch.load(VAL_DATA_PATH)
    test_set = torch.load(TEST_DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(
            {
                "model_type": "Transformer",
                "learning_rate": LR,
                "batch_size": BS,
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "loss_type": LOSS_TYPE,
                "seed": seed,
                "leave_ratio": LEAVE_RATIO,
                "data_type": DATA_TYPE,
                "smoothing_type": smoothing_type,
                "kernel_size": KERNEL_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "n_time_series": N_TIME_SERIES,
                "feature_dim": FEATURE_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "stability_period": stability_period,
                "initial_alpha": initial_alpha,
                "final_alpha": final_alpha,
            }
        )
        if LOSS_TYPE == "Cauchy":
            mlflow.log_param("cauchy_loss_gamma", CAUCHY_LOSS_GAMMA)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TransformerWithPE(
            NUM_FEATURES, NUM_FEATURES, FEATURE_DIM, NUM_HEADS, NUM_LAYERS
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = make_criterion(LOSS_TYPE, CAUCHY_LOSS_GAMMA)

        train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            split_name="synthetic",
            num_epoch=NUM_EPOCHS,
            device=device,
        )

        logger.info("Evaluating model on held-out test set")
        evaluate_model(
            model=model,
            criterion=criterion,
            loader=test_loader,
            split_name="test",
            device=device,
        )

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
