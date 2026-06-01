import logging
import os

import click
import matplotlib.pyplot as plt
import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BS, FIGURES_DIR, NUM_VIS_EXAMPLES, TEST_DATA_PATH, TRACKING_URI
from src.models.utils import error_metrics, model_forward, prepare_batch
from src.visualization.visualize import visualize_prediction

EXPERIMENT_NAME = "initial experiments"


def resolve_run_id(client, experiment_name, run_id):
    """Return an explicit run id, or the most recent run of the experiment."""
    if run_id:
        return run_id

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise click.ClickException(f'MLflow experiment "{experiment_name}" not found.')

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise click.ClickException(f'No runs found for experiment "{experiment_name}".')
    return runs[0].info.run_id


def _mean(dicts):
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


@click.command()
@click.option(
    "--run-id", default=None, help="MLflow run id to load the model from (default: latest run)."
)
@click.option(
    "--experiment-name", default=EXPERIMENT_NAME, help="Experiment to search for the latest run."
)
@click.option("--data-path", default=TEST_DATA_PATH, help="Path to the .pt dataset to predict on.")
@click.option(
    "--num-examples",
    default=NUM_VIS_EXAMPLES,
    type=int,
    help="Number of example predictions to plot.",
)
def main(run_id, experiment_name, data_path, num_examples):
    """Load a trained model from MLflow and forecast on a held-out dataset.

    Works for both architectures (LSTM and Transformer) - the split and forward
    pass are selected automatically from the loaded model. Reports AUTOREGRESSIVE
    metrics as the primary result and teacher-forced metrics as an optimistic
    reference.
    """
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    run_id = resolve_run_id(client, experiment_name, run_id)
    logger.info(f"Loading model from run {run_id}")

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info(f"Loading dataset from {data_path}")
    dataset = torch.load(data_path)
    loader = DataLoader(dataset, batch_size=BS, shuffle=False)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    tf_metrics, ar_metrics = [], []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Predicting", unit="batch")):
            src, tgt, tgt_y = prepare_batch(batch[0], device)

            pred_tf = model_forward(model, src, tgt, tgt_y)
            tf_metrics.append(error_metrics(pred_tf, tgt_y))

            pred_infer = model.infer(src, tgt_y.shape[1])
            ar_metrics.append(error_metrics(pred_infer, tgt_y))

            if idx < num_examples:
                figure = visualize_prediction(src, tgt_y, pred_tf, pred_infer)
                figure.savefig(os.path.join(FIGURES_DIR, f"prediction_{idx}.png"))
                plt.close(figure)

    ar = _mean(ar_metrics)
    tf = _mean(tf_metrics)
    logger.info(
        f"Autoregressive (primary) - MSE: {ar['mse']:.4f}, MAE: {ar['mae']:.4f}, "
        f"RMSE: {ar['rmse']:.4f}, MAPE: {ar['mape']:.2f}, SMAPE: {ar['smape']:.2f}"
    )
    logger.info(
        f"Teacher-forced (optimistic) - MSE: {tf['mse']:.4f}, MAE: {tf['mae']:.4f}, "
        f"RMSE: {tf['rmse']:.4f}, MAPE: {tf['mape']:.2f}, SMAPE: {tf['smape']:.2f}"
    )
    logger.info(f"Saved up to {num_examples} prediction plots to {FIGURES_DIR}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
