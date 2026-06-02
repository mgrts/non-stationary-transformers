"""Orchestrates the full experiment sweep.

Structure (methodologically important):
  1. Download the real OWID data ONCE up front (cached, idempotent) so the
     "real" dataset does not silently change between conditions.
  2. For each DATA condition (alpha / stability / smoothing grid) build the
     dataset ONCE: generate -> process -> make_dataset (train/val/test split).
  3. For each model SEED, train on that fixed dataset. Repeating over seeds
     with the data held constant isolates model variance from data/split noise,
     enabling mean/std reporting.

A failure in a data-prep step aborts that condition's chain (later steps depend
on it); the sweep continues with the next condition and exits non-zero if any
condition or run failed.
"""

import logging
import os
import subprocess
import sys
from itertools import product

from src.config import (
    FINAL_ALPHAS,
    INITIAL_ALPHAS,
    PROJECT_ROOT_DIR,
    SEEDS,
    SMOOTHING_TYPES,
    STABILITY_PERIODS,
)

logger = logging.getLogger(__name__)


def run_step(script_rel, args, check=True):
    """Run a pipeline script as a subprocess; raise on failure when check=True."""
    script = os.path.join(PROJECT_ROOT_DIR, script_rel)
    command = [sys.executable, script] + args
    logger.info(f'Running {script_rel} {" ".join(args)}')
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = f"{script_rel} failed (exit {result.returncode}):\n{result.stderr}"
        if check:
            raise RuntimeError(message)
        logger.error(message)
        return False
    logger.info(f"Completed {script_rel}.\n{result.stdout}")
    return True


def build_dataset(stability_period, initial_alpha, final_alpha, smoothing_type):
    """Generate + process + split one data condition (download handled once, earlier)."""
    common_data_args = [
        "--stability-period",
        stability_period,
        "--initial-alpha",
        str(initial_alpha),
        "--final-alpha",
        str(final_alpha),
    ]
    run_step("src/data/generate_data.py", common_data_args)
    run_step("src/data/process_data.py", ["--smoothing-type", smoothing_type])
    run_step("src/data/make_dataset.py", [])


def train_over_seeds(seeds):
    """Train the model once per seed on the already-built dataset.

    The data condition (alphas / stability / smoothing) is read by train_model
    from the data_meta.json sidecar written during build_dataset, so only the
    model seed varies here - the logged condition can never drift from the data.
    """
    for seed in seeds:
        run_step("src/models/train_model.py", ["--seed", str(seed)])


def main():
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # 1. Download real data once (idempotent; skips if already cached).
    run_step("src/data/download_data.py", [])

    # 2-3. Sweep data conditions, training over seeds for each.
    failures = []
    conditions = list(product(INITIAL_ALPHAS, FINAL_ALPHAS, STABILITY_PERIODS, SMOOTHING_TYPES))
    logger.info(f"Sweeping {len(conditions)} data conditions x {len(SEEDS)} seeds each.")

    for initial_alpha, final_alpha, stability_period, smoothing_type in conditions:
        condition = dict(
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            stability_period=stability_period,
            smoothing_type=smoothing_type,
        )
        try:
            build_dataset(stability_period, initial_alpha, final_alpha, smoothing_type)
            train_over_seeds(SEEDS)
        except Exception as exc:
            logger.error(f"Condition failed {condition}: {exc}")
            failures.append(condition)

    if failures:
        logger.error(f"{len(failures)}/{len(conditions)} conditions failed: {failures}")
        sys.exit(1)
    logger.info("All conditions completed successfully.")


if __name__ == "__main__":
    main()
