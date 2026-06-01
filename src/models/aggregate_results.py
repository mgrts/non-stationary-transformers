"""Aggregate multi-seed sweep results into mean +/- std per data condition.

The training pipeline trains each data condition once per seed (see
src/training_pipeline.py and config.SEEDS). Each run is logged to MLflow
separately; this script pulls those runs back, groups them by the data-defining
parameters, and reports the mean and standard deviation of the test metrics
across seeds - which is the point of the multi-seed design (variance estimation).

The AUTOREGRESSIVE test metrics (prefix ``test_ar_``) are the primary result.
"""

import logging
import statistics
from collections import defaultdict

import click
import mlflow
from mlflow.tracking import MlflowClient

from src.config import TRACKING_URI

# Parameters that DEFINE a data condition (runs sharing these differ only by seed).
CONDITION_PARAMS = ("initial_alpha", "final_alpha", "stability_period", "smoothing_type")
# Metric suffixes aggregated for a split (autoregressive = primary performance).
DEFAULT_METRIC_SUFFIXES = ("ar_mse", "ar_mae", "ar_rmse", "ar_smape")
# Default split prefix. train_model.py logs `test_*`; train_model_real.py logs
# `synthetic_test_*` / `real_test_*` - pass --split to aggregate those.
DEFAULT_SPLIT = "test"


def metric_keys_for_split(split, suffixes=DEFAULT_METRIC_SUFFIXES):
    """Build the logged metric keys for a split, e.g. 'test' -> 'test_ar_mse'."""
    return tuple(f"{split}_{suffix}" for suffix in suffixes)


def group_runs(runs):
    """Group MLflow runs by their data-condition parameter tuple."""
    groups = defaultdict(list)
    for run in runs:
        key = tuple(run.data.params.get(p, "NA") for p in CONDITION_PARAMS)
        groups[key].append(run)
    return groups


def aggregate(values):
    """Return (mean, std, n); std is the sample std (NA for a single run)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return mean, std, n


@click.command()
@click.option(
    "--experiment-name", default="initial experiments", help="MLflow experiment to aggregate."
)
@click.option(
    "--split",
    default=DEFAULT_SPLIT,
    help="Split prefix of the metric keys to aggregate (e.g. test, real_test, synthetic_test).",
)
@click.option(
    "--metrics",
    default=None,
    help="Comma-separated metric keys to aggregate (overrides --split if given).",
)
def main(experiment_name, split, metrics):
    logger = logging.getLogger(__name__)
    if metrics:
        metric_keys = [m.strip() for m in metrics.split(",") if m.strip()]
    else:
        metric_keys = list(metric_keys_for_split(split))

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise click.ClickException(f'Experiment "{experiment_name}" not found.')

    runs = client.search_runs([experiment.experiment_id], max_results=50000)
    if not runs:
        raise click.ClickException(f'No runs found for experiment "{experiment_name}".')

    groups = group_runs(runs)
    logger.info(
        f"Aggregating {len(runs)} runs across {len(groups)} data conditions "
        f'(experiment="{experiment_name}").'
    )

    for key in sorted(groups):
        condition = dict(zip(CONDITION_PARAMS, key))
        runs_in_group = groups[key]
        print(f"\n=== {condition} ({len(runs_in_group)} seeds) ===")
        for metric in metric_keys:
            values = [r.data.metrics[metric] for r in runs_in_group if metric in r.data.metrics]
            mean, std, n = aggregate(values)
            if n == 0:
                print(f"  {metric}: (not logged)")
            else:
                print(f"  {metric}: {mean:.4f} +/- {std:.4f}  (n={n})")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
