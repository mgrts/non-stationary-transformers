import logging

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from src.config import (INTERIM_DATA_PATH, OWID_INTERIM_DATA_PATH,
                        OWID_TEST_DATA_PATH, OWID_TRAIN_DATA_PATH,
                        SYNTHETIC_COVID_INTERIM_DATA_PATH,
                        SYNTHETIC_COVID_TEST_DATA_PATH,
                        SYNTHETIC_COVID_TRAIN_DATA_PATH, TEST_DATA_PATH,
                        TRAIN_DATA_PATH)


def make_datasets(sequences: np.ndarray) -> tuple[TensorDataset, TensorDataset]:
    """Create train and test dataset.

    Args:
        sequences: sequences to use [num_sequences, sequence_length, num_features]

    Returns:
        tuple[TensorDataset, TensorDataset]: train and test dataset
    """
    # Split sequences into train and test split
    train, test = train_test_split(sequences, test_size=0.2)
    return TensorDataset(torch.Tensor(train)), TensorDataset(torch.Tensor(test))


def main():
    logger = logging.getLogger(__name__)

    logger.info('Preparing datasets')

    for data_path in (INTERIM_DATA_PATH, OWID_INTERIM_DATA_PATH, SYNTHETIC_COVID_INTERIM_DATA_PATH):
        with open(data_path, 'rb') as f:
            sequences = np.load(f)

        train_set, test_set = make_datasets(sequences)

        if data_path == INTERIM_DATA_PATH:
            train_path = TRAIN_DATA_PATH
            test_path = TEST_DATA_PATH
        elif data_path == OWID_INTERIM_DATA_PATH:
            train_path = OWID_TRAIN_DATA_PATH
            test_path = OWID_TEST_DATA_PATH
        elif data_path == SYNTHETIC_COVID_INTERIM_DATA_PATH:
            train_path = SYNTHETIC_COVID_TRAIN_DATA_PATH
            test_path = SYNTHETIC_COVID_TEST_DATA_PATH

        torch.save(train_set, train_path)
        torch.save(test_set, test_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
