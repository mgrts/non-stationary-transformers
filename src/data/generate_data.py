import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
# import click
from scipy.stats import cauchy

from src.config import (DATA_TYPE, FINAL_SCALE, INITIAL_FRAC_BOUNDS,
                        INITIAL_SCALE, INTERVAL, N_TIME_SERIES, RANDOM_STATE,
                        RAW_DATA_PATH, SIGMA, TIME_SERIES_LENGTH,
                        TRANSITION_FRAC_BOUNDS)


def generate_non_stationary_time_series(length, initial_scale, final_scale, initial_frac, transition_frac, sigma, type):
    initial_length = int(length * initial_frac)
    transition_length = int(length * transition_frac)
    final_length = length - initial_length - transition_length

    scales = np.concatenate([
        np.repeat(initial_scale, initial_length),
        np.linspace(initial_scale, final_scale, transition_length),
        np.repeat(final_scale, final_length)
    ])

    if type == 'cauchy':
        time_series = cauchy.rvs(loc=0, scale=scales, size=length)
        time_series = gaussian_filter1d(time_series, sigma=sigma)
    elif type == 'sine':
        x = np.linspace(0, length * INTERVAL, length)
        y = np.sin(x) + np.random.normal(0, 0.1, x.shape)
        time_series = y * scales
    else:
        raise AttributeError(f'Type {type} is not supported.')

    return time_series


def generate_data(n, length, initial_scale, final_scale, initial_frac_bounds, transition_frac_bounds, sigma):
    num_features = 1
    sequences = np.empty((n, length, num_features))

    initial_fracs = np.random.uniform(low=initial_frac_bounds[0], high=initial_frac_bounds[1], size=n)
    transition_fracs = np.random.uniform(low=transition_frac_bounds[0], high=transition_frac_bounds[1], size=n)

    for i, initial_frac, transition_frac in zip(range(n), initial_fracs, transition_fracs):
        # [sequence_length, num_features]
        ts = generate_non_stationary_time_series(
            length=length,
            initial_scale=initial_scale,
            final_scale=final_scale,
            initial_frac=initial_frac,
            transition_frac=transition_frac,
            sigma=sigma,
            type=DATA_TYPE
        )
        sample = np.asarray([ts]).swapaxes(0, 1)
        sequences[i] = sample

    return sequences


# @click.command()
# @click.argument('length', type=int)
# @click.argument('initial_scale', type=float)
# @click.argument('final_scale', type=float)
# @click.option('--seed', type=int, default=None, help="Random seed for reproducibility (optional).")
def main():
    """ Generates a non-stationary time series with varying scale.
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating data')

    if RANDOM_STATE is not None:
        np.random.seed(RANDOM_STATE)

    data = generate_data(
        n=N_TIME_SERIES,
        length=TIME_SERIES_LENGTH,
        initial_scale=INITIAL_SCALE,
        final_scale=FINAL_SCALE,
        initial_frac_bounds=INITIAL_FRAC_BOUNDS,
        transition_frac_bounds=TRANSITION_FRAC_BOUNDS,
        sigma=SIGMA
    )

    logger.info('Saving data')

    with open(RAW_DATA_PATH, 'wb') as f:
        np.save(f, data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
