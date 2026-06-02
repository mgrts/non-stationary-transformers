non-stationary-transformers
==============================

Non-stationary time series prediction using transformer neural networks

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make verify`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── pyproject.toml     <- Project metadata, dependencies, and tool config (Poetry / PEP 621).
    ├── poetry.lock        <- Pinned, resolved dependency versions (committed for reproducibility).
    ├── src                <- Source code (full module list + descriptions in CLAUDE.md).
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Reference
https://github.com/hermanmichaels/transformer_example
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levy_stable.html
https://www.cs.toronto.edu/~duvenaud/cookbook

## Set up project

```bash
poetry install            # runtime + dev tooling into an in-project .venv
cp .env.example .env       # then edit values as needed (see src/config.py)
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Run any command inside the environment with `poetry run <cmd>` (e.g.
`poetry run python src/training_pipeline.py`) or open a shell with `poetry shell`.

## Training pipeline

Synthetic transformer pipeline:

1. `src/data/download_data.py` - download the real OWID COVID CSV once (cached, idempotent)
2. `src/data/generate_data.py` - generate non-stationary and synthetic-COVID series
3. `src/data/process_data.py` - smooth + **causally** normalise into interim data
4. `src/data/make_dataset.py` - build **train/val/test** `TensorDataset`s (group-aware for OWID)
5. `src/models/train_model.py` - train the transformer (early stopping on validation)
6. `src/models/predict_model.py` - load a trained model from MLflow and forecast

Or run the whole sweep at once with `python src/training_pipeline.py`, which
downloads once, builds each data condition once, and trains across `SEEDS` for
variance estimation. Afterwards, `python src/models/aggregate_results.py`
collects the per-seed MLflow runs and prints mean +/- std of the test metrics
per data condition.

Real-world (COVID) experiment: `src/models/train_model_real.py` pre-trains on the
synthetic data and fine-tunes on the OWID dataset (`--model-type LSTM|Transformer`).

### Methodology notes

- **No look-ahead leakage:** normalization statistics are fit on each sequence's
  history window only (`causal_normalize`), never the forecast horizon.
- **Group-aware real split:** OWID chunks are split by country, so no country
  spans train/val/test; chunks are non-overlapping.
- **Honest evaluation:** the primary metric is the *autoregressive* forecast
  (model consumes its own outputs); teacher-forced numbers are reported only as
  an optimistic reference. Validation/early-stopping also use the autoregressive
  loss. Metrics include MSE/MAE/RMSE alongside MAPE/SMAPE.
- **Metric scale:** all metrics are computed on the *normalized* scale (the
  per-sequence causal scaler is not retained for inverse-transforming), so
  MSE/MAE/RMSE are comparable across models on the same dataset but not in
  original units. MAPE is masked to `|target| > eps` and is only a rough
  secondary indicator on this ~0-centered data; prefer MSE/MAE/RMSE/SMAPE.
- **Reproducibility:** `RANDOM_STATE` fixes data generation and the split;
  `SEEDS` varies only model initialisation/training so reported variance is
  model variance. See `src/seeding.py`.

All hyperparameters are read from the environment (copy `.env.example` to `.env`).
Shared model/training helpers live in `src/models/model.py` and `src/models/utils.py`.
