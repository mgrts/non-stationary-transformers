import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.config import (INTERIM_DATA_PATH, KERNEL_SIZE, RAW_DATA_PATH, SIGMA,
                        SMOOTHING_TYPE)


def sine_kernel(size):
    """Generates a normalized sine-based kernel for smoothing"""
    assert size % 2 == 1, "Size must be odd."
    x = np.linspace(0, np.pi, size)
    kernel = np.sin(x)
    kernel /= np.sum(kernel)
    return kernel


def sine_filter1d(data, size=9):
    """Applies sine-based smoothing to a 1D numpy array"""
    kernel = sine_kernel(size)
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data


def smooth_sequence(sequence, smoothing_type, kernel_size):
    sequence_initial_shape = sequence.shape

    sequence = sequence.reshape(-1)

    if smoothing_type == 'gaussian':
        sequence = gaussian_filter1d(sequence, radius=(int((kernel_size - 1) / 2)), sigma=SIGMA)
    elif smoothing_type == 'sine':
        sequence = sine_filter1d(sequence, size=kernel_size)
    else:
        raise AttributeError(f'Smoothing type {smoothing_type} is not supported.')

    sequence = sequence.reshape(sequence_initial_shape)

    return sequence


def main():
    logger = logging.getLogger(__name__)

    logger.info('Processing data')

    with open(RAW_DATA_PATH, 'rb') as f:
        sequences = np.load(f)

    processed_sequences = np.empty(sequences.shape)

    for i in range(len(sequences)):
        processed_sequence = smooth_sequence(sequences[i], smoothing_type=SMOOTHING_TYPE, kernel_size=KERNEL_SIZE)
        processed_sequences[i] = processed_sequence

    logger.info('Saving processed data')

    with open(INTERIM_DATA_PATH, 'wb') as f:
        np.save(f, processed_sequences)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
