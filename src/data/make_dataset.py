import logging
import os

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import TensorDataset

from src.config import (
    INTERIM_DATA_PATH,
    OWID_GROUPS_PATH,
    OWID_INTERIM_DATA_PATH,
    OWID_TEST_DATA_PATH,
    OWID_TRAIN_DATA_PATH,
    OWID_VAL_DATA_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SYNTHETIC_COVID_INTERIM_DATA_PATH,
    SYNTHETIC_COVID_TEST_DATA_PATH,
    SYNTHETIC_COVID_TRAIN_DATA_PATH,
    SYNTHETIC_COVID_VAL_DATA_PATH,
    TEST_DATA_PATH,
    TEST_RATIO,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    VAL_RATIO,
)
from src.seeding import seed_everything


def random_three_way_split(sequences, val_ratio, test_ratio, seed):
    """Random (i.i.d.) train/val/test split for independently generated series."""
    train_val, test = train_test_split(sequences, test_size=test_ratio, random_state=seed)
    # val_ratio is expressed relative to the FULL set, so rescale for the remainder.
    val_relative = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed)
    return train, val, test


def grouped_three_way_split(sequences, groups, val_ratio, test_ratio, seed):
    """Group-aware split: a group (e.g. a country) never spans two splits.

    This prevents leakage where chunks from the same country - which share
    regime, scale and dynamics - appear in both train and test, which would make
    the test score an over-optimistic estimate of generalization to new series.
    """
    groups = np.asarray(groups)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(sequences, groups=groups))

    tv_sequences = sequences[train_val_idx]
    tv_groups = groups[train_val_idx]
    val_relative = val_ratio / (1.0 - test_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(gss_val.split(tv_sequences, groups=tv_groups))

    return tv_sequences[train_idx], tv_sequences[val_idx], sequences[test_idx]


def to_datasets(train, val, test):
    return (
        TensorDataset(torch.Tensor(train)),
        TensorDataset(torch.Tensor(val)),
        TensorDataset(torch.Tensor(test)),
    )


def main():
    logger = logging.getLogger(__name__)
    logger.info("Preparing datasets")

    # Fixed seed: the data split is held constant across model-training seeds so
    # the multi-seed sweep measures model variance, not split variance.
    seed_everything(RANDOM_STATE)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # (interim path, (train, val, test) output paths, groups path or None)
    datasets = [
        (INTERIM_DATA_PATH, (TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH), None),
        (
            SYNTHETIC_COVID_INTERIM_DATA_PATH,
            (
                SYNTHETIC_COVID_TRAIN_DATA_PATH,
                SYNTHETIC_COVID_VAL_DATA_PATH,
                SYNTHETIC_COVID_TEST_DATA_PATH,
            ),
            None,
        ),
        (
            OWID_INTERIM_DATA_PATH,
            (OWID_TRAIN_DATA_PATH, OWID_VAL_DATA_PATH, OWID_TEST_DATA_PATH),
            OWID_GROUPS_PATH,
        ),
    ]

    for data_path, (train_path, val_path, test_path), groups_path in datasets:
        with open(data_path, "rb") as f:
            sequences = np.load(f)

        if groups_path is not None:
            with open(groups_path, "rb") as f:
                groups = np.load(f)
            # The group-leakage guarantee relies on groups[i] being the country
            # of sequences[i]. Fail loudly if the two interim artifacts ever
            # drift out of sync (e.g. a stale groups file) instead of silently
            # mislabelling chunks and leaking a country across splits.
            assert len(groups) == len(sequences), (
                f"OWID groups/sequences length mismatch: "
                f"{len(groups)} groups vs {len(sequences)} sequences. "
                f"Re-run process_data.py to regenerate both interim artifacts."
            )
            train, val, test = grouped_three_way_split(
                sequences, groups, VAL_RATIO, TEST_RATIO, RANDOM_STATE
            )
            logger.info(
                f"{os.path.basename(data_path)}: grouped split -> "
                f"train={len(train)}, val={len(val)}, test={len(test)} "
                f"({len(np.unique(groups))} groups)"
            )
        else:
            train, val, test = random_three_way_split(
                sequences, VAL_RATIO, TEST_RATIO, RANDOM_STATE
            )
            logger.info(
                f"{os.path.basename(data_path)}: random split -> "
                f"train={len(train)}, val={len(val)}, test={len(test)}"
            )

        train_set, val_set, test_set = to_datasets(train, val, test)
        torch.save(train_set, train_path)
        torch.save(val_set, val_path)
        torch.save(test_set, test_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
