import logging
import numpy as np
import click


@click.command()
@click.argument('length', type=int)
@click.argument('initial_variance', type=float)
@click.argument('final_variance', type=float)
@click.option('--seed', type=int, default=None, help="Random seed for reproducibility (optional).")
def main(length, initial_variance, final_variance, seed):
    """ Generates a non-stationary time series with varying variance and plots it.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if seed is not None:
        np.random.seed(seed)

        # Create a linearly increasing or decreasing variance over the specified length
    variances = np.linspace(initial_variance, final_variance, length)

    # Generate the time series
    time_series = np.random.normal(loc=0.0, scale=np.sqrt(variances), size=length)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
