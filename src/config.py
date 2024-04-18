import os

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT_DIR = os.getenv('PROJECT_ROOT_DIR')

RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/data.npy')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/processed/data.npy')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')


RANDOM_STATE = 927

DATA_TYPE = 'cauchy'
# DATA_TYPE = 'sine'
TIME_SERIES_LENGTH = 200
N_TIME_SERIES = 10000
INITIAL_SCALE = 0.5 if DATA_TYPE == 'cauchy' else 1
FINAL_SCALE = 2 if DATA_TYPE == 'cauchy' else 3
INITIAL_FRAC_BOUNDS = (0.3, 0.5)
TRANSITION_FRAC_BOUNDS = (0.1, 0.3)
SIGMA = 8
INTERVAL = 0.1

# BS = 512
BS = 128
FEATURE_DIM = 128
NUM_HEADS = 8
# NUM_EPOCHS = 100
NUM_EPOCHS = 20
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.001
