import logging

import click
import numpy as np
from scipy.stats import levy_stable

from src.config import (DATA_TYPE, FINAL_ALPHA, INITIAL_ALPHA,
                        INITIAL_FRAC_BOUNDS_LONG, INITIAL_FRAC_BOUNDS_MODERATE,
                        INITIAL_FRAC_BOUNDS_SHORT, N_TIME_SERIES, NUM_FEATURES,
                        RANDOM_STATE, RAW_DATA_PATH, SEQUENCE_LENGTH,
                        SINE_INTERVAL, STABILITY_PERIOD,
                        TRANSITION_FRAC_BOUNDS_LONG,
                        TRANSITION_FRAC_BOUNDS_MODERATE,
                        TRANSITION_FRAC_BOUNDS_SHORT)


def generate_non_stationary_sequence(length, initial_alpha, final_alpha, initial_frac, transition_frac, type):
    initial_length = int(length * initial_frac)
    transition_length = int(length * transition_frac)
    final_length = length - initial_length - transition_length

    alphas = np.concatenate([
        np.repeat(initial_alpha, initial_length),
        np.linspace(initial_alpha, final_alpha, transition_length),
        np.repeat(final_alpha, final_length)
    ])

    if type == 'random':
        sequence = levy_stable.rvs(alpha=alphas, beta=0)
    elif type == 'sine':
        x = np.linspace(0, length * SINE_INTERVAL, length)
        y = np.sin(x) + np.random.normal(0, 0.1, x.shape)
        sequence = y * alphas

    return sequence


def generate_data(n, length, initial_alpha, final_alpha, initial_frac_bounds, transition_frac_bounds):
    sequences = np.empty((n, length, NUM_FEATURES))

    initial_fracs = np.random.uniform(low=initial_frac_bounds[0], high=initial_frac_bounds[1], size=n)
    transition_fracs = np.random.uniform(low=transition_frac_bounds[0], high=transition_frac_bounds[1], size=n)

    for i, initial_frac, transition_frac in zip(range(n), initial_fracs, transition_fracs):
        # [sequence_length, num_features]
        ts = generate_non_stationary_sequence(
            length=length,
            initial_alpha=initial_alpha,
            final_alpha=final_alpha,
            initial_frac=initial_frac,
            transition_frac=transition_frac,
            type=DATA_TYPE
        )
        sample = np.asarray([ts]).swapaxes(0, 1)
        sequences[i] = sample

    return sequences


@click.command()
@click.option('--stability-period', default=STABILITY_PERIOD,
              type=click.Choice(['short', 'moderate', 'long'], case_sensitive=False),
              help='Period of stability (short, moderate, long)')
@click.option('--initial-alpha', default=INITIAL_ALPHA, type=float, help='Initial alpha value for generating sequences')
@click.option('--final-alpha', default=FINAL_ALPHA, type=float, help='Final alpha value for generating sequences')
def main(stability_period, initial_alpha, final_alpha):
    """ Generates a non-stationary sequence with varying stability.
    """
    logger = logging.getLogger(__name__)

    logger.info('Generating data')

    if RANDOM_STATE is not None:
        np.random.seed(RANDOM_STATE)

    if stability_period == 'short':
        initial_frac_bounds = INITIAL_FRAC_BOUNDS_SHORT
        transition_frac_bounds = TRANSITION_FRAC_BOUNDS_SHORT
    elif stability_period == 'moderate':
        initial_frac_bounds = INITIAL_FRAC_BOUNDS_MODERATE
        transition_frac_bounds = TRANSITION_FRAC_BOUNDS_MODERATE
    elif stability_period == 'long':
        initial_frac_bounds = INITIAL_FRAC_BOUNDS_LONG
        transition_frac_bounds = TRANSITION_FRAC_BOUNDS_LONG
    else:
        raise AttributeError(f'Stability period {stability_period} is not supported')

    data = generate_data(
        n=N_TIME_SERIES,
        length=SEQUENCE_LENGTH,
        initial_alpha=initial_alpha,
        final_alpha=final_alpha,
        initial_frac_bounds=initial_frac_bounds,
        transition_frac_bounds=transition_frac_bounds
    )

    logger.info('Saving data')

    with open(RAW_DATA_PATH, 'wb') as f:
        np.save(f, data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
