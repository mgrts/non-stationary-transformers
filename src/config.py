import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT_DIR = os.getenv('PROJECT_ROOT_DIR')
LOSS_TYPE = os.getenv('LOSS_TYPE')
DATA_TYPE = os.getenv('DATA_TYPE')
SMOOTHING_TYPE = os.getenv('SMOOTHING_TYPE')
STABILITY_PERIOD = os.getenv('STABILITY_PERIOD')
KERNEL_SIZE = int(os.getenv('KERNEL_SIZE'))
SIGMA = int(os.getenv('SIGMA'))
PERIOD = int(os.getenv('PERIOD'))
SEQUENCE_LENGTH = int(os.getenv('SEQUENCE_LENGTH'))
N_TIME_SERIES = int(os.getenv('N_TIME_SERIES'))
SINE_INTERVAL = float(os.getenv('SINE_INTERVAL'))
CAUCHY_LOSS_GAMMA = float(os.getenv('CAUCHY_LOSS_GAMMA'))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/data.npy')
INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/interim/data.npy')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data/processed')
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
TRACKING_URI = os.path.join(PROJECT_ROOT_DIR, 'mlruns')

SIR_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/sir_data.npy')

SYNTHETIC_COVID_RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/synthetic_covid_data.npy')
SYNTHETIC_COVID_INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/interim/synthetic_covid_data.npy')

OWID_DATA_URL = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
OWID_RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/raw/owid_data.csv')
OWID_INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'data/interim/owid_data.npy')

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_data.pt')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_data.pt')
OWID_TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'owid_train_data.pt')
OWID_TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'owid_test_data.pt')
SYNTHETIC_COVID_TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'synthetic_covid_train_data.pt')
SYNTHETIC_COVID_TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'synthetic_covid_test_data.pt')

RANDOM_STATE = 927

INITIAL_ALPHA = 2 if DATA_TYPE == 'random' else 1
FINAL_ALPHA = 1.8 if DATA_TYPE == 'random' else 3

INITIAL_FRAC_BOUNDS_SHORT = (0.05, 0.1)
TRANSITION_FRAC_BOUNDS_SHORT = (0.05, 0.1)
INITIAL_FRAC_BOUNDS_LONG = (0.65, 0.75)
TRANSITION_FRAC_BOUNDS_LONG = (0.05, 0.1)
INITIAL_FRAC_BOUNDS_MODERATE = (0.3, 0.4)
TRANSITION_FRAC_BOUNDS_MODERATE = (0.05, 0.1)

NUM_FEATURES = 1
BS = int(os.getenv('BS'))
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.001

COVID_SEQ_CHUNK_SIZE = 300
