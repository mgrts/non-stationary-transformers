"""Reusable time-series feature transforms for non-stationary sequences.

These are importable helpers rather than a mandatory pipeline step: the core
modelling pipeline (generate -> process -> make_dataset -> train -> predict)
consumes the raw smoothed series directly. The functions here make it easy to
engineer additional stationarity-oriented features (differencing, returns,
rolling statistics) for experimentation.

Unless noted otherwise, functions operate along the time axis (axis=1) of a
batch of sequences shaped ``[num_sequences, seq_len]`` or
``[num_sequences, seq_len, 1]``.
"""

import logging

import numpy as np

from src.config import INTERIM_DATA_PATH


def _as_2d(sequences):
    """Collapse a ``[n, seq, 1]`` batch to ``[n, seq]``; pass ``[n, seq]`` through."""
    sequences = np.asarray(sequences, dtype=float)
    if sequences.ndim == 3 and sequences.shape[-1] == 1:
        return sequences[..., 0]
    if sequences.ndim == 2:
        return sequences
    raise ValueError(
        "Expected sequences shaped [n, seq] or [n, seq, 1], " f"got {sequences.shape}."
    )


def difference(sequences, order=1):
    """N-th order differencing along time to remove trends (improves stationarity).

    Left-padded with zeros so the output keeps the input length.
    """
    data = _as_2d(sequences)
    diffed = np.diff(data, n=order, axis=1)
    pad = np.zeros((data.shape[0], order))
    return np.concatenate([pad, diffed], axis=1)


def log_returns(sequences, eps=1e-8):
    """Log returns ``log(x_t / x_{t-1})`` along time, left-padded with zeros.

    Uses the magnitude plus a small epsilon so it stays defined for
    non-positive values.
    """
    data = _as_2d(sequences)
    magnitude = np.abs(data) + eps
    returns = np.log(magnitude[:, 1:] / magnitude[:, :-1])
    pad = np.zeros((data.shape[0], 1))
    return np.concatenate([pad, returns], axis=1)


def _rolling(sequences, window, reduce_fn):
    if window < 1:
        raise ValueError("window must be >= 1")
    data = _as_2d(sequences)
    _, seq_len = data.shape
    out = np.zeros_like(data, dtype=float)
    for t in range(seq_len):
        start = max(0, t - window + 1)
        out[:, t] = reduce_fn(data[:, start : t + 1], axis=1)
    return out


def rolling_mean(sequences, window):
    """Causal rolling mean over a trailing window of length ``window``."""
    return _rolling(sequences, window, np.mean)


def rolling_std(sequences, window):
    """Causal rolling standard deviation over a trailing window of length ``window``."""
    return _rolling(sequences, window, np.std)


def sliding_windows(series, window_size, step=1):
    """Turn a 1D series into overlapping windows shaped ``[num_windows, window_size]``."""
    series = np.asarray(series).reshape(-1)
    if window_size > len(series):
        raise ValueError("window_size cannot exceed the series length")
    starts = range(0, len(series) - window_size + 1, step)
    return np.stack([series[s : s + window_size] for s in starts])


def build_feature_stack(sequences, rolling_window=7):
    """Stack the raw series with engineered features into ``[n, seq, num_features]``.

    Channels: ``[raw, first-difference, log-returns, rolling-mean, rolling-std]``.
    """
    data = _as_2d(sequences)
    features = [
        data,
        difference(data),
        log_returns(data),
        rolling_mean(data, rolling_window),
        rolling_std(data, rolling_window),
    ]
    return np.stack(features, axis=-1)


def main():
    logger = logging.getLogger(__name__)
    logger.info("Building example feature stack from interim data")

    with open(INTERIM_DATA_PATH, "rb") as f:
        sequences = np.load(f)

    stacked = build_feature_stack(sequences)
    logger.info(
        f"Input {np.asarray(sequences).shape} -> feature stack {stacked.shape} "
        "(raw, diff, log-returns, rolling-mean, rolling-std)"
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
