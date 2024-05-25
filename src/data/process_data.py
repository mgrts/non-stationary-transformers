import logging

import click
import numpy as np

from src.config import (INTERIM_DATA_PATH, KERNEL_SIZE, PERIOD, RAW_DATA_PATH,
                        SIGMA, SMOOTHING_TYPE)


def safe_normalize(kernel):
    """Safely normalizes a kernel to prevent division by zero"""
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError("The sum of the kernel weights is zero, which makes normalization impossible.")
    else:
        kernel /= kernel_sum
    return kernel


def sine_kernel(size, period):
    """Generates a normalized sine-based kernel for smoothing"""
    x = np.linspace(0, 2 * np.pi * size / period, size)
    kernel = (np.sin(x) + 1) / 2
    kernel = safe_normalize(kernel)
    return kernel


def cosine_kernel(size, period):
    """Generate a cosine kernel with adjustable period."""
    # Establish the range of indices
    num_cycles = size / period

    # Generate a cosine kernel
    x = np.linspace(0, 2 * np.pi * num_cycles, size)
    kernel = (np.cos(x) + 1) / 2
    kernel = safe_normalize(kernel)  # Normalize the kernel
    return kernel


def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    # Establish the range of values
    kernel_range = np.arange(-size // 2 + 1., size // 2 + 1.)
    # Calculate the Gaussian function for each point in the kernel range
    kernel = np.exp(-0.5 * (kernel_range / sigma) ** 2)
    kernel = safe_normalize(kernel)  # Normalize the kernel to ensure the sum is 1
    return kernel


def combined_cosine_gaussian_kernel(size, sigma, period):
    """Generates a combined cosine and Gaussian kernel for smoothing"""
    cosine_k = cosine_kernel(size, period)
    gaussian_k = gaussian_kernel(size, sigma)
    combined_kernel = cosine_k * gaussian_k
    combined_kernel = safe_normalize(combined_kernel)
    return combined_kernel


def filter1d_with_kernel(data, kernel):
    """Applies smoothing to a 1D numpy array using a given kernel"""
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data


def smooth_sequence(sequence, smoothing_type, kernel_size, sigma, period):
    sequence_initial_shape = sequence.shape
    sequence = sequence.reshape(-1)

    if smoothing_type == 'gaussian':
        kernel = gaussian_kernel(kernel_size, sigma)
    elif smoothing_type == 'sine':
        kernel = sine_kernel(kernel_size, period)
    elif smoothing_type == 'cosine':
        kernel = cosine_kernel(kernel_size, period)
    elif smoothing_type == 'combined_cosine_gaussian':
        kernel = combined_cosine_gaussian_kernel(kernel_size, sigma, period)
    else:
        raise AttributeError(f'Smoothing type {smoothing_type} is not supported.')

    sequence = filter1d_with_kernel(sequence, kernel)
    sequence = sequence.reshape(sequence_initial_shape)
    return sequence


@click.command()
@click.option('--smoothing-type', default=SMOOTHING_TYPE,
              help='Type of smoothing to apply (gaussian, sine, cosine, combined_cosine_gaussian)')
def main(smoothing_type):
    logger = logging.getLogger(__name__)
    logger.info('Processing data')

    with open(RAW_DATA_PATH, 'rb') as f:
        sequences = np.load(f)

    processed_sequences = np.empty(sequences.shape)

    for i in range(len(sequences)):
        processed_sequence = smooth_sequence(sequence=sequences[i], smoothing_type=smoothing_type,
                                             kernel_size=KERNEL_SIZE, sigma=SIGMA, period=PERIOD)
        processed_sequences[i] = processed_sequence

    logger.info('Saving processed data')
    with open(INTERIM_DATA_PATH, 'wb') as f:
        np.save(f, processed_sequences)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
