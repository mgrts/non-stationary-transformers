import os

from dotenv import load_dotenv

load_dotenv()


def _env_list(name, default, cast):
    """Parse a comma-separated env var into a list, applying `cast` per item."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


# All settings can be overridden through environment variables (e.g. via a
# local .env file - see .env.example). Sensible defaults are provided so the
# project is importable and runnable out of the box without a populated .env.
# PROJECT_ROOT_DIR falls back to the repository root inferred from this file's
# location (src/config.py -> repo root is two levels up).
PROJECT_ROOT_DIR = os.getenv("PROJECT_ROOT_DIR") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
LOSS_TYPE = os.getenv("LOSS_TYPE", "MSE")
DATA_TYPE = os.getenv("DATA_TYPE", "random")
SMOOTHING_TYPE = os.getenv("SMOOTHING_TYPE", "gaussian")
STABILITY_PERIOD = os.getenv("STABILITY_PERIOD", "moderate")
KERNEL_SIZE = int(os.getenv("KERNEL_SIZE", "25"))
SIGMA = int(os.getenv("SIGMA", "5"))
PERIOD = int(os.getenv("PERIOD", "50"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "300"))
N_TIME_SERIES = int(os.getenv("N_TIME_SERIES", "1000"))
SINE_INTERVAL = float(os.getenv("SINE_INTERVAL", "0.1"))
CAUCHY_LOSS_GAMMA = float(os.getenv("CAUCHY_LOSS_GAMMA", "1.0"))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data/raw/data.npy")
INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data/interim/data.npy")
# Sidecar JSON recording the data-generation condition (alphas, stability, smoothing)
# so trainers log the TRUE condition the data was built with, instead of re-reading
# their own CLI flags (which can drift from how the data was actually generated).
DATA_META_PATH = os.path.join(PROJECT_ROOT_DIR, "data/interim/data_meta.json")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data/processed")
MODELS_DIR = os.path.join(PROJECT_ROOT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TRACKING_URI = os.path.join(PROJECT_ROOT_DIR, "mlruns")

# MLflow >= 3.x puts the filesystem tracking backend (./mlruns) in maintenance
# mode and raises unless this opt-in is set. This project intentionally uses the
# local file store, so enable it by default (overridable via the environment).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

SYNTHETIC_COVID_RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data/raw/synthetic_covid_data.npy")
SYNTHETIC_COVID_INTERIM_DATA_PATH = os.path.join(
    PROJECT_ROOT_DIR, "data/interim/synthetic_covid_data.npy"
)

OWID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
OWID_RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data/raw/owid_data.csv")
OWID_INTERIM_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data/interim/owid_data.npy")
# Per-sequence country labels for the OWID chunks, used to split by group so a
# country never spans train/val/test (avoids leakage).
OWID_GROUPS_PATH = os.path.join(PROJECT_ROOT_DIR, "data/interim/owid_groups.npy")

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_data.pt")
VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "val_data.pt")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_data.pt")
OWID_TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "owid_train_data.pt")
OWID_VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "owid_val_data.pt")
OWID_TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "owid_test_data.pt")
SYNTHETIC_COVID_TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "synthetic_covid_train_data.pt")
SYNTHETIC_COVID_VAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "synthetic_covid_val_data.pt")
SYNTHETIC_COVID_TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "synthetic_covid_test_data.pt")

# Seed used for *data generation and the train/val/test split* - kept fixed so
# the dataset and splits are identical across model-training seeds (the
# multi-seed sweep then measures model variance, not data/split noise).
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "927"))

INITIAL_ALPHA = 2 if DATA_TYPE == "random" else 1
FINAL_ALPHA = 1.8 if DATA_TYPE == "random" else 3

INITIAL_FRAC_BOUNDS_SHORT = (0.05, 0.1)
TRANSITION_FRAC_BOUNDS_SHORT = (0.05, 0.1)
INITIAL_FRAC_BOUNDS_LONG = (0.65, 0.75)
TRANSITION_FRAC_BOUNDS_LONG = (0.05, 0.1)
INITIAL_FRAC_BOUNDS_MODERATE = (0.3, 0.4)
TRANSITION_FRAC_BOUNDS_MODERATE = (0.05, 0.1)

# --- Model / training hyperparameters (all env-overridable and logged) -------
NUM_FEATURES = 1
BS = int(os.getenv("BS", "32"))
# Batch size for the (smaller) real-world OWID dataset in train_model_real.py.
REAL_BS = int(os.getenv("REAL_BS", "16"))
# Fraction of each sequence fed to the encoder; the remainder is the forecast
# target. Used by the shared train/eval split helpers.
LEAVE_RATIO = float(os.getenv("LEAVE_RATIO", "0.8"))
FEATURE_DIM = int(os.getenv("FEATURE_DIM", "128"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "8"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
NUM_VIS_EXAMPLES = int(os.getenv("NUM_VIS_EXAMPLES", "10"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "2"))
LR = float(os.getenv("LR", "0.001"))
# Early-stopping patience (epochs without val improvement) for the trainers.
PATIENCE = int(os.getenv("PATIENCE", "5"))

COVID_SEQ_CHUNK_SIZE = int(os.getenv("COVID_SEQ_CHUNK_SIZE", "300"))

# --- Data split ---------------------------------------------------------------
# Three-way split fractions (train = 1 - VAL_RATIO - TEST_RATIO).
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.1"))

# --- Normalization ------------------------------------------------------------
# Causal (history-window-fit) scaler applied to every dataset. 'robust' uses
# median/IQR (resistant to the heavy tails of levy-stable data); 'standard'
# uses mean/std. Count-like data (COVID) is log1p-transformed first.
NORMALIZATION_SCALER = os.getenv("NORMALIZATION_SCALER", "robust")

# --- Experiment sweep (training_pipeline.py) ----------------------------------
# Data-defining grid (each combination is generated/processed/split once).
INITIAL_ALPHAS = _env_list("INITIAL_ALPHAS", [2.0], float)
FINAL_ALPHAS = _env_list("FINAL_ALPHAS", [2.0, 1.9, 1.5], float)
STABILITY_PERIODS = _env_list("STABILITY_PERIODS", ["short", "moderate", "long"], str)
SMOOTHING_TYPES = _env_list("SMOOTHING_TYPES", ["gaussian", "combined_cosine_gaussian"], str)
# Model-training seeds: each data condition is trained once per seed so we can
# report mean/std across seeds (variance estimation).
SEEDS = _env_list("SEEDS", [927, 123, 2024], int)
