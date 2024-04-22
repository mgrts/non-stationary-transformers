import logging
import os
import subprocess

from src.config import PROJECT_ROOT_DIR


def run_python_files(file_list):
    for file in file_list:
        print(f'Running {file}...')
        result = subprocess.run(['python', file], capture_output=True, text=True)
        if result.returncode == 0:
            print(f'Completed {file} successfully!\nOutput:\n{result.stdout}')
        else:
            print(f'Error in {file}:\n{result.stderr}')


def main():
    logger = logging.getLogger(__name__)

    logger.info('Running training pipeline')

    files_to_run = [
        os.path.join(PROJECT_ROOT_DIR, 'src/data/generate_data.py'),
        os.path.join(PROJECT_ROOT_DIR, 'src/data/process_data.py'),
        os.path.join(PROJECT_ROOT_DIR, 'src/data/make_dataset.py'),
        os.path.join(PROJECT_ROOT_DIR, 'src/models/train_model.py'),
    ]

    run_python_files(files_to_run)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
