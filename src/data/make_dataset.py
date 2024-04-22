import logging

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from src.config import INTERIM_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH


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

    logger.info('Preparing dataset')

    with open(INTERIM_DATA_PATH, 'rb') as f:
        sequences = np.load(f)

    train_set, test_set = make_datasets(sequences)

    logger.info('Saving dataset')

    torch.save(train_set, TRAIN_DATA_PATH)
    torch.save(test_set, TEST_DATA_PATH)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
