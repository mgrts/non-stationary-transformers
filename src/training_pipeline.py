import logging
import os
import subprocess
from itertools import product

import click

from src.config import PROJECT_ROOT_DIR


def run_python_files(file_list, stability_period, initial_alpha, final_alpha, smoothing_type):
    for file in file_list:
        command = ['python', file]

        if 'generate_data.py' in file:
            command.extend(
                ['--stability-period', stability_period, '--initial-alpha', str(initial_alpha), '--final-alpha',
                 str(final_alpha)])
        elif 'process_data.py' in file:
            command.extend(['--smoothing-type', smoothing_type])
        elif 'train_model.py' in file:
            command.extend(
                ['--smoothing-type', smoothing_type, '--stability-period', stability_period, '--initial-alpha',
                 str(initial_alpha), '--final-alpha',
                 str(final_alpha)])

        print(f'Running {file} with parameters: {" ".join(command[2:])}...')
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f'Completed {file} successfully!\nOutput:\n{result.stdout}')
        else:
            print(f'Error in {file}:\n{result.stderr}')


@click.command()
@click.option('--stability-period', default='moderate', help='Period of stability (short, moderate, long)')
@click.option('--initial-alpha', default=1.5, type=float, help='Initial alpha value for generating sequences')
@click.option('--final-alpha', default=1.8, type=float, help='Final alpha value for generating sequences')
@click.option('--smoothing-type', default='gaussian', help='Type of smoothing to apply (e.g., gaussian, sine)')
def main(stability_period, initial_alpha, final_alpha, smoothing_type):
    logger = logging.getLogger(__name__)
    logger.info('Running training pipeline')

    files_to_run = [
        os.path.join(PROJECT_ROOT_DIR, 'src/data/generate_data.py'),
        os.path.join(PROJECT_ROOT_DIR, 'src/data/process_data.py'),
        os.path.join(PROJECT_ROOT_DIR, 'src/data/make_dataset.py'),
        os.path.join(PROJECT_ROOT_DIR, 'src/models/train_model.py'),
    ]

    run_python_files(files_to_run, stability_period, initial_alpha, final_alpha, smoothing_type)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    initial_alphas = [2]
    final_alphas = [2, 1.9, 1.5]
    stability_periods = ['short', 'moderate', 'long']
    smoothing_types = ['gaussian', 'combined_cosine_gaussian']

    for initial_alpha, final_alpha, stability_period, smoothing_type in product(initial_alphas, final_alphas,
                                                                                stability_periods, smoothing_types):
        ctx = click.Context(main, info_name='main', parent=None)
        ctx.invoke(main, stability_period=stability_period, initial_alpha=initial_alpha,
                   final_alpha=final_alpha, smoothing_type=smoothing_type)
