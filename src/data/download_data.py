"""Idempotent download of the real-world OWID COVID dataset.

Separated from data generation so the (large, slowly-changing) download happens
once and is cached, rather than being re-fetched on every experiment iteration.
The OWID feed is updated over time, so re-downloading mid-study would silently
change the "real" dataset; run this once and reuse the cached CSV for
reproducibility (use --force to deliberately refresh).
"""

import logging
import os

import click
import requests
from tqdm import tqdm

from src.config import OWID_DATA_URL, OWID_RAW_DATA_PATH


def download_with_progress(url, path, chunk_size=1024):
    """Stream `url` to `path` with a progress bar."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(path, "wb") as file:
        for data in response.iter_content(chunk_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        raise IOError(f"Download incomplete: expected {total_size} bytes, got {t.n}.")
    return path


@click.command()
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even if the cached file already exists.",
)
def main(force):
    logger = logging.getLogger(__name__)

    if os.path.exists(OWID_RAW_DATA_PATH) and not force:
        logger.info(
            f"OWID data already present at {OWID_RAW_DATA_PATH}; "
            "skipping download (use --force to refresh)."
        )
        return

    logger.info(f"Downloading real COVID data from {OWID_DATA_URL}")
    download_with_progress(OWID_DATA_URL, OWID_RAW_DATA_PATH)
    logger.info(f"Saved OWID data to {OWID_RAW_DATA_PATH}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
