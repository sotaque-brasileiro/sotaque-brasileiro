from os import environ
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from sotaque_brasileiro.constants import constants
from sotaque_brasileiro.io import (
    fetch_paginated_data,
    download_multiple,
    load_envs_from_file,
)
from sotaque_brasileiro.preprocessing import webm_to_wav

__all__ = [
    "get_updated_dataframe",
    "download_dataset",
]


def safe_getenv(key: str) -> str:
    """
    Get an environment variable safely,
    tries to load it from file if not found.

    Args:
        key: The key of the environment variable.

    Returns:
        The value of the environment variable, if exists.

    Raises:
        KeyError: If the key does not exist.
    """
    try:
        return environ[key]
    except KeyError:
        env_file = Path(constants.ENV_FILE_DEFAULT_PATH.value)
        if env_file.exists() and env_file.is_file():
            try:
                load_envs_from_file()
                return environ[key]
            except KeyError:
                raise KeyError(
                    f"{key} is not set and the default env file was not found."
                )


def parse_records_to_dataframe(
    records: List[Dict], timezone: str = constants.TIMEZONE_DEFAULT.value
) -> pd.DataFrame:
    """
    Parse a list of records into a pandas dataframe.

    Args:
        records: List of records to parse.

    Returns:
        A pandas dataframe.
    """
    # Get initial dataframe
    df = pd.DataFrame(records)

    # Extract id from url
    df["id"] = df["url"].apply(lambda x: int(x.split("/")[-2]))
    df.drop("url", axis=1, inplace=True)

    # Get data from sentence and remove it from the dataframe
    df["sentence_text"] = df["sentence"].apply(lambda x: x["text"])
    df.drop("sentence", axis=1, inplace=True)

    # Get data from speaker and remove it from the dataframe
    df["gender"] = df["speaker"].apply(lambda x: x["gender"]["name"])
    df["age"] = df["speaker"].apply(lambda x: x["age"])
    df["profession"] = df["speaker"].apply(lambda x: x["profession"])
    df["birth_city"] = df["speaker"].apply(lambda x: x["birth_city"]["name"])
    df["birth_state"] = df["speaker"].apply(
        lambda x: x["birth_city"]["state"]["abbreviation"]
    )
    df["birth_latitude"] = df["speaker"].apply(lambda x: x["birth_city"]["latitude"])
    df["birth_longitude"] = df["speaker"].apply(lambda x: x["birth_city"]["longitude"])
    df["current_city"] = df["speaker"].apply(lambda x: x["current_city"]["name"])
    df["current_state"] = df["speaker"].apply(
        lambda x: x["current_city"]["state"]["abbreviation"]
    )
    df["current_latitude"] = df["speaker"].apply(
        lambda x: x["current_city"]["latitude"]
    )
    df["current_longitude"] = df["speaker"].apply(
        lambda x: x["current_city"]["longitude"]
    )
    df["years_on_current_city"] = df["speaker"].apply(
        lambda x: x["years_on_current_city"]
    )
    df["parents_original_city"] = df["speaker"].apply(
        lambda x: x["parents_original_city"]["name"]
    )
    df["parents_original_state"] = df["speaker"].apply(
        lambda x: x["parents_original_city"]["state"]["abbreviation"]
    )
    df["parents_original_latitude"] = df["speaker"].apply(
        lambda x: x["parents_original_city"]["latitude"]
    )
    df["parents_original_longitude"] = df["speaker"].apply(
        lambda x: x["parents_original_city"]["longitude"]
    )
    df.drop("speaker", axis=1, inplace=True)

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.tz_convert(timezone)

    # Order columns
    df = df[
        [
            "id",
            "date",
            "sentence_text",
            "audio_file_path",
            "gender",
            "age",
            "profession",
            "birth_city",
            "birth_state",
            "birth_latitude",
            "birth_longitude",
            "current_city",
            "current_state",
            "current_latitude",
            "current_longitude",
            "years_on_current_city",
            "parents_original_city",
            "parents_original_state",
            "parents_original_latitude",
            "parents_original_longitude",
        ]
    ]

    return df


def get_updated_dataframe():
    """
    Get the dataframe with the latest data from the API.
    """
    records = fetch_paginated_data(constants.API_RECORDS_ENDPOINT.value)
    df = parse_records_to_dataframe(records)
    return df


def download_dataset(
    path_to_save: str = constants.DATASET_SAVE_DEFAULT_PATH.value,
    show_progress: bool = constants.SHOW_PROGRESS_DEFAULT.value,
):
    """
    Download an updated version of the dataset from the API and MinIO.
    """
    print("Fetching updated dataframe...")
    df = get_updated_dataframe()
    save_dir = Path(path_to_save)
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(save_dir / "sotaque-brasileiro.csv"), index=False)
    blob_list = list(df["audio_file_path"])
    df["audio_file_path"] = df["audio_file_path"].apply(lambda x: str(save_dir / x))
    file_list = list(df["audio_file_path"])
    if not show_progress:
        print("Downloading audio files...")
    download_multiple(
        bucket_name=safe_getenv(constants.MINIO_BUCKET.value),
        object_names=blob_list,
        file_paths=file_list,
        show_progress=show_progress,
    )
    if show_progress:
        if tqdm is None:
            raise ImportError(
                "tqdm must be installed to show progress. Either install tqdm or run this command with show_progress=False"
            )
        for file in tqdm(file_list, desc="Converting audio files..."):
            webm_to_wav(file)
            tqdm.write(file)
    else:
        print("Converting audio files...")
        for file in file_list:
            webm_to_wav(file)
