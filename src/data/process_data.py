import logging

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (COVID_SEQ_CHUNK_SIZE, INTERIM_DATA_PATH, KERNEL_SIZE,
                        OWID_INTERIM_DATA_PATH, OWID_RAW_DATA_PATH, PERIOD,
                        RAW_DATA_PATH, SIGMA, SMOOTHING_TYPE,
                        SYNTHETIC_COVID_INTERIM_DATA_PATH,
                        SYNTHETIC_COVID_RAW_DATA_PATH)


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


def slice_array_to_chunks(array, chunk_size=300):
    n = len(array)
    n_slices = (n + chunk_size - 1) // chunk_size  # ceil(n / chunk_size)
    chunks = []

    for i in range(n_slices):
        start = i * chunk_size
        end = start + chunk_size
        if end > n:
            end = n
            start = max(0, n - chunk_size)
        chunks.append(array[start:end])

    return np.array(chunks)


@click.command()
@click.option('--smoothing-type', default=SMOOTHING_TYPE,
              help='Type of smoothing to apply (gaussian, sine, cosine, combined_cosine_gaussian)')
def main(smoothing_type):
    logger = logging.getLogger(__name__)

    logger.info('Processing data with different stability periods')

    with open(RAW_DATA_PATH, 'rb') as f:
        sequences = np.load(f)

    processed_sequences = np.empty(sequences.shape)

    for i in range(len(sequences)):
        processed_sequence = smooth_sequence(sequence=sequences[i], smoothing_type=smoothing_type,
                                             kernel_size=KERNEL_SIZE, sigma=SIGMA, period=PERIOD)
        processed_sequences[i] = processed_sequence

    logger.info('Processing synthetic COVID data')

    with open(SYNTHETIC_COVID_RAW_DATA_PATH, 'rb') as f:
        synthetic_covid_sequences = np.load(f)

    smoothed_synthetic_sequences = []

    for seq in synthetic_covid_sequences:
        smoothed_sequence = smooth_sequence(
            sequence=seq, smoothing_type='gaussian', kernel_size=KERNEL_SIZE, sigma=SIGMA, period=PERIOD
        )
        smoothed_sequence = StandardScaler().fit_transform(smoothed_sequence.reshape(-1, 1)).reshape(-1)
        smoothed_synthetic_sequences.append(smoothed_sequence)

    smoothed_synthetic_sequences = np.array(smoothed_synthetic_sequences)
    smoothed_synthetic_sequences = smoothed_synthetic_sequences[..., np.newaxis]

    logger.info('Processing real COVID data')

    covid_data = pd.read_csv(OWID_RAW_DATA_PATH)

    countries = [
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
        'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
        'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'United States', 'Russia',
        'Ukraine', 'Belarus', 'Kazakhstan', 'Armenia', 'Azerbaijan', 'Georgia', 'Kyrgyzstan', 'Moldova',
        'Tajikistan', 'Turkmenistan', 'Uzbekistan'
    ]

    # Filter data for the selected countries
    covid_data = covid_data[covid_data['location'].isin(countries)]

    # Select relevant columns and handle missing values
    covid_data = covid_data[['location', 'date', 'new_cases']]
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    covid_data = covid_data.sort_values(['location', 'date'])
    covid_data['new_cases'] = covid_data['new_cases'].fillna(0)  # Fill missing values with 0

    # Apply smoothing to remove zeros (simple moving average)
    covid_data['new_cases'] = covid_data.groupby('location')['new_cases'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Pivot the data to have dates as rows and locations as columns
    pivot_data = covid_data.pivot(index='date', columns='location', values='new_cases')
    pivot_data = pivot_data.fillna(0)  # Ensure no missing values remain

    owid_sequences = []
    for column_name in pivot_data.columns:
        column = pivot_data[column_name]
        chunks = slice_array_to_chunks(column, COVID_SEQ_CHUNK_SIZE)

        for chunk in chunks:
            chunk_scaled = StandardScaler().fit_transform(chunk.reshape(-1, 1)).reshape(-1)
            owid_sequences.append(chunk_scaled)

    owid_sequences = np.vstack(owid_sequences)
    owid_sequences = owid_sequences[..., np.newaxis]

    logger.info('Saving processed data')

    with open(INTERIM_DATA_PATH, 'wb') as f:
        np.save(f, processed_sequences)

    with open(SYNTHETIC_COVID_INTERIM_DATA_PATH, 'wb') as f:
        np.save(f, smoothed_synthetic_sequences)

    with open(OWID_INTERIM_DATA_PATH, 'wb') as f:
        np.save(f, owid_sequences)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
