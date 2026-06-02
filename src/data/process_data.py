import json
import logging
import os

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import (
    COVID_SEQ_CHUNK_SIZE,
    DATA_META_PATH,
    INTERIM_DATA_PATH,
    KERNEL_SIZE,
    LEAVE_RATIO,
    NORMALIZATION_SCALER,
    OWID_GROUPS_PATH,
    OWID_INTERIM_DATA_PATH,
    OWID_RAW_DATA_PATH,
    PERIOD,
    RAW_DATA_PATH,
    SIGMA,
    SMOOTHING_TYPE,
    SYNTHETIC_COVID_INTERIM_DATA_PATH,
    SYNTHETIC_COVID_RAW_DATA_PATH,
)


def make_scaler(kind=NORMALIZATION_SCALER):
    """Return a fresh scaler. 'robust' (median/IQR) resists heavy tails."""
    if kind == "robust":
        return RobustScaler()
    elif kind == "standard":
        return StandardScaler()
    raise ValueError(f"Normalization scaler {kind} is not supported.")


def causal_normalize(sequence, leave_ratio=LEAVE_RATIO, kind=NORMALIZATION_SCALER, log1p=False):
    """Normalize a 1D sequence using statistics from its history window only.

    The scaler is fit on the first ``leave_ratio`` fraction of the sequence (the
    encoder/history window) and applied to the whole series. This prevents the
    forecast horizon from leaking into the inputs via the normalization
    constants. ``log1p`` first applies a log transform for non-negative count
    data (e.g. COVID cases) to compress its skew. sklearn scalers map a
    zero-variance window to scale 1.0, so constant histories are safe.
    """
    seq = np.asarray(sequence, dtype=float).reshape(-1, 1)
    if log1p:
        seq = np.log1p(np.clip(seq, a_min=0, a_max=None))
    split = max(1, int(len(seq) * leave_ratio))
    scaler = make_scaler(kind)
    scaler.fit(seq[:split])
    return scaler.transform(seq).reshape(-1)


def safe_normalize(kernel):
    """Safely normalizes a kernel to prevent division by zero"""
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError(
            "The sum of the kernel weights is zero, which makes normalization impossible."
        )
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
    kernel_range = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
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
    smoothed_data = np.convolve(data, kernel, mode="same")
    return smoothed_data


def smooth_sequence(sequence, smoothing_type, kernel_size, sigma, period):
    sequence_initial_shape = sequence.shape
    sequence = sequence.reshape(-1)

    if smoothing_type == "gaussian":
        kernel = gaussian_kernel(kernel_size, sigma)
    elif smoothing_type == "sine":
        kernel = sine_kernel(kernel_size, period)
    elif smoothing_type == "cosine":
        kernel = cosine_kernel(kernel_size, period)
    elif smoothing_type == "combined_cosine_gaussian":
        kernel = combined_cosine_gaussian_kernel(kernel_size, sigma, period)
    else:
        raise ValueError(f"Smoothing type {smoothing_type} is not supported.")

    sequence = filter1d_with_kernel(sequence, kernel)
    sequence = sequence.reshape(sequence_initial_shape)
    return sequence


def slice_array_to_chunks(array, chunk_size=300):
    """Split a series into NON-overlapping chunks of exactly `chunk_size`.

    The trailing partial chunk is dropped (not back-extended), so chunks never
    overlap - overlapping chunks would leak data between train/val/test when one
    chunk lands in train and its overlapping neighbour in test.

    Returns:
        tuple[list[np.ndarray], int]: equal-length chunks and the number of
        trailing samples dropped.
    """
    array = np.asarray(array)
    n = len(array)
    n_full = n // chunk_size
    chunks = [array[i * chunk_size : (i + 1) * chunk_size] for i in range(n_full)]
    dropped = n - n_full * chunk_size
    return chunks, dropped


@click.command()
@click.option(
    "--smoothing-type",
    default=SMOOTHING_TYPE,
    help="Type of smoothing to apply (gaussian, sine, cosine, combined_cosine_gaussian)",
)
def main(smoothing_type):
    logger = logging.getLogger(__name__)

    logger.info("Processing data with different stability periods")

    os.makedirs(os.path.dirname(INTERIM_DATA_PATH), exist_ok=True)

    with open(RAW_DATA_PATH, "rb") as f:
        sequences = np.load(f)

    processed_sequences = np.empty(sequences.shape)

    for i in range(len(sequences)):
        smoothed = smooth_sequence(
            sequence=sequences[i],
            smoothing_type=smoothing_type,
            kernel_size=KERNEL_SIZE,
            sigma=SIGMA,
            period=PERIOD,
        )
        # Causal robust normalization (no log1p: levy-stable data can be negative).
        normalized = causal_normalize(smoothed.reshape(-1), log1p=False)
        processed_sequences[i] = normalized.reshape(smoothed.shape)

    logger.info("Processing synthetic COVID data")

    with open(SYNTHETIC_COVID_RAW_DATA_PATH, "rb") as f:
        synthetic_covid_sequences = np.load(f)

    smoothed_synthetic_sequences = []

    for seq in synthetic_covid_sequences:
        smoothed_sequence = smooth_sequence(
            sequence=seq,
            smoothing_type="gaussian",
            kernel_size=KERNEL_SIZE,
            sigma=SIGMA,
            period=PERIOD,
        )
        # Counts are non-negative -> log1p before causal normalization.
        smoothed_sequence = causal_normalize(smoothed_sequence.reshape(-1), log1p=True)
        smoothed_synthetic_sequences.append(smoothed_sequence)

    smoothed_synthetic_sequences = np.array(smoothed_synthetic_sequences)
    smoothed_synthetic_sequences = smoothed_synthetic_sequences[..., np.newaxis]

    logger.info("Processing real COVID data")

    covid_data = pd.read_csv(OWID_RAW_DATA_PATH)

    countries = [
        "Austria",
        "Belgium",
        "Bulgaria",
        "Croatia",
        "Cyprus",
        "Czechia",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Ireland",
        "Italy",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "Poland",
        "Portugal",
        "Romania",
        "Slovakia",
        "Slovenia",
        "Spain",
        "Sweden",
        "United States",
        "Russia",
        "Ukraine",
        "Belarus",
        "Kazakhstan",
        "Armenia",
        "Azerbaijan",
        "Georgia",
        "Kyrgyzstan",
        "Moldova",
        "Tajikistan",
        "Turkmenistan",
        "Uzbekistan",
    ]

    # Filter data for the selected countries
    covid_data = covid_data[covid_data["location"].isin(countries)]

    # Select relevant columns and handle missing values
    covid_data = covid_data[["location", "date", "new_cases"]]
    covid_data["date"] = pd.to_datetime(covid_data["date"])
    covid_data = covid_data.sort_values(["location", "date"])
    covid_data["new_cases"] = covid_data["new_cases"].fillna(0)  # Fill missing values with 0

    # Apply smoothing to remove zeros (simple moving average)
    covid_data["new_cases"] = covid_data.groupby("location")["new_cases"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Pivot the data to have dates as rows and locations as columns
    pivot_data = covid_data.pivot(index="date", columns="location", values="new_cases")
    pivot_data = pivot_data.fillna(0)  # Ensure no missing values remain

    owid_sequences = []
    owid_groups = []  # country index per chunk, for group-aware splitting
    total_dropped = 0
    for country_idx, column_name in enumerate(pivot_data.columns):
        column = pivot_data[column_name].to_numpy()
        chunks, dropped = slice_array_to_chunks(column, COVID_SEQ_CHUNK_SIZE)
        total_dropped += dropped
        for chunk in chunks:
            chunk_scaled = causal_normalize(chunk, log1p=True)
            owid_sequences.append(chunk_scaled)
            owid_groups.append(country_idx)

    if not owid_sequences:
        raise ValueError(
            f"No OWID chunks of length {COVID_SEQ_CHUNK_SIZE} could be built; "
            "every country series is shorter than one chunk."
        )

    owid_sequences = np.vstack(owid_sequences)[..., np.newaxis]
    owid_groups = np.array(owid_groups)
    logger.info(
        f"OWID: built {len(owid_sequences)} non-overlapping chunks across "
        f"{pivot_data.shape[1]} countries; dropped {total_dropped} trailing days total"
    )

    logger.info("Saving processed data")

    with open(INTERIM_DATA_PATH, "wb") as f:
        np.save(f, processed_sequences)

    with open(SYNTHETIC_COVID_INTERIM_DATA_PATH, "wb") as f:
        np.save(f, smoothed_synthetic_sequences)

    with open(OWID_INTERIM_DATA_PATH, "wb") as f:
        np.save(f, owid_sequences)

    with open(OWID_GROUPS_PATH, "wb") as f:
        np.save(f, owid_groups)

    # Enrich the data-generation metadata with the smoothing actually applied, so
    # trainers can log the TRUE condition (alphas + stability from generate_data,
    # smoothing from here) rather than re-reading their own CLI flags.
    meta = {}
    if os.path.exists(DATA_META_PATH):
        with open(DATA_META_PATH) as f:
            meta = json.load(f)
    meta["smoothing_type"] = smoothing_type
    with open(DATA_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
