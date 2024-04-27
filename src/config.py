import os

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT_DIR = os.getenv('PROJECT_ROOT_DIR')
LOSS_TYPE = os.getenv('LOSS_TYPE')
DATA_TYPE = os.getenv('DATA_TYPE')
SMOOTHING_TYPE = os.getenv('SMOOTHING_TYPE')
STABILITY_PERIOD = os.getenv('STABILITY_PERIOD')
KERNEL_SIZE = int(os.getenv('KERNEL_SIZE'))
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH'))
N_TIME_SERIES = int(os.getenv('N_TIME_SERIES'))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/data.npy')
INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/interim/data.npy')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data/processed')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_data.pt')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_data.pt')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
TRACKING_URI = os.path.join(PROJECT_ROOT_DIR, 'mlruns')

RANDOM_STATE = 927

INITIAL_ALPHA = 1.8 if DATA_TYPE == 'random' else 1
FINAL_ALPHA = 1 if DATA_TYPE == 'random' else 3

if STABILITY_PERIOD == 'short':
    INITIAL_FRAC_BOUNDS = (0.05, 0.1)
    TRANSITION_FRAC_BOUNDS = (0.05, 0.1)
elif STABILITY_PERIOD == 'long':
    INITIAL_FRAC_BOUNDS = (0.65, 0.75)
    TRANSITION_FRAC_BOUNDS = (0.05, 0.1)
elif STABILITY_PERIOD == 'moderate':
    INITIAL_FRAC_BOUNDS = (0.3, 0.4)
    TRANSITION_FRAC_BOUNDS = (0.05, 0.1)
else:
    raise AttributeError(f'Stability period {STABILITY_PERIOD} is not supported')

SIGMA = 8
SINE_INTERVAL = 0.1

NUM_FEATURES = 1
BS = int(os.getenv('BS'))
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.001
