import os

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT_DIR = os.getenv('PROJECT_ROOT_DIR')

RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/data.npy')
INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/interim/data.npy')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data/processed')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_data.pt')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_data.pt')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')


RANDOM_STATE = 927

DATA_TYPE = 'random'
# DATA_TYPE = 'sine'
SMOOTHING_TYPE = 'gaussian'
# SMOOTHING_TYPE = 'sine'
SEQUENCE_LENGTH = 300
N_TIME_SERIES = 10000
INITIAL_ALPHA = 1.8 if DATA_TYPE == 'random' else 1
FINAL_ALPHA = 1 if DATA_TYPE == 'random' else 3
INITIAL_FRAC_BOUNDS = (0.3, 0.5)
TRANSITION_FRAC_BOUNDS = (0.1, 0.3)
SIGMA = 8
SINE_INTERVAL = 0.1
KERNEL_SIZE = 51

NUM_FEATURES = 1
# BS = 512
BS = 128
FEATURE_DIM = 128
NUM_HEADS = 8
# NUM_EPOCHS = 100
NUM_EPOCHS = 20
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.001
