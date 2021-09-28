"""
General purpose utilities functions.
"""
from os import environ
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from p_tqdm import p_map

from sotaque_brasileiro.constants import constants
from sotaque_brasileiro.io import safe_getenv, load_multiple_wav
from sotaque_brasileiro import preprocessing, feature
from sotaque_brasileiro.preprocessing import resample, stereo_to_mono

__all__ = [
    "get_updated_dataframe",
    "download_dataset",
]


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
    df = pd.DataFrame(records)  # pylint: disable=invalid-name

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
    df["birth_latitude"] = df["speaker"].apply(
        lambda x: x["birth_city"]["latitude"])
    df["birth_longitude"] = df["speaker"].apply(
        lambda x: x["birth_city"]["longitude"])
    df["current_city"] = df["speaker"].apply(
        lambda x: x["current_city"]["name"])
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
    # pylint: disable=invalid-name
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
    # pylint: disable=import-outside-toplevel
    from sotaque_brasileiro.io import fetch_paginated_data
    records = fetch_paginated_data(constants.API_RECORDS_ENDPOINT.value)
    df = parse_records_to_dataframe(records)  # pylint: disable=invalid-name
    return df


def download_dataset(
    path_to_save: str = constants.DATASET_SAVE_DEFAULT_PATH.value,
    show_progress: bool = constants.SHOW_PROGRESS_DEFAULT.value,
):
    """
    Download an updated version of the dataset from the API and GCS.
    """
    # pylint: disable=import-outside-toplevel
    from sotaque_brasileiro.io import download_multiple, webm_to_wav
    print("Fetching updated dataframe...")
    df = get_updated_dataframe()  # pylint: disable=invalid-name
    save_dir = Path(path_to_save)
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(save_dir / "sotaque-brasileiro.csv"), index=False)
    blob_list = list(df["audio_file_path"])
    df["audio_file_path"] = df["audio_file_path"].apply(
        lambda x: str(save_dir / x))
    file_list = list(df["audio_file_path"])
    if not show_progress:
        print("Downloading audio files...")
    download_multiple(
        bucket_name=safe_getenv(constants.GCS_BUCKET_NAME.value),
        object_names=blob_list,
        file_paths=file_list,
        show_progress=show_progress,
    )
    if show_progress:
        if tqdm is None:
            raise ImportError(
                """tqdm must be installed to show progress.
                Either install tqdm or run this command with show_progress=False"""
            )
        for file in tqdm(file_list, desc="Converting audio files..."):
            webm_to_wav(file)
            tqdm.write(file)
    else:
        print("Converting audio files...")
        for file in file_list:
            webm_to_wav(file)


def load_all_audios(
    df: pd.DataFrame,
    *,
    target_sample_rate: int = None,
    mono: bool = constants.STEREO_TO_MONO_DEFAULT.value
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Load all audios from the dataframe.
    """
    file_list = list(df["audio_file_path"])

    # audios is a list of (rate: int, data: np.ndarray)
    audios = load_multiple_wav(file_list)
    rate = [i[0] for i in audios]
    data = [i[1] for i in audios]

    # Convert to mono if needed
    if mono:
        data = p_map(stereo_to_mono, data, desc="Converting to mono...")

    # Resample if needed
    if target_sample_rate is not None:
        data = p_map(resample, data, rate, [
                     target_sample_rate for _ in data], desc="Resampling...")
        rate = [target_sample_rate for _ in data]

    return rate, data


def apply_preprocessing(
    data: List[np.ndarray],
    sample_rate: Union[int, List[int]],
    funcs: List[str],
    *,
    kwargs: dict = {},
) -> np.ndarray:
    """
    Apply preprocessing to the data.
    `funcs` is a list of functions to apply.
    The function `resample` is always applied first.
    """
    if isinstance(sample_rate, int):
        sample_rate = [sample_rate] * len(data)
    try:
        assert len(data) == len(sample_rate)
    except AssertionError:
        raise ValueError(
            "The number of sample rates must be the same as the number of data"
        )
    for func in funcs:
        if hasattr(preprocessing, func):
            preproc: Callable = getattr(preprocessing, func, **kwargs)
            data = p_map(preproc, data, sample_rate,
                         desc=f"Applying {func}...")
        else:
            raise ValueError(f"{func} is not a valid preprocessing function")
    return data


def extract_features(
    frames: List[np.ndarray],
    sample_rate: Union[int, List[int]],
    funcs: List[str],
    *,
    kwargs: dict = {},
) -> Dict[str, np.ndarray]:
    """
    Extract features from the data.
    `funcs` is a list of functions to apply.
    """
    if isinstance(sample_rate, int):
        sample_rate = [sample_rate] * len(frames)
    try:
        assert len(frames) == len(sample_rate)
    except AssertionError:
        raise ValueError(
            "The number of sample rates must be the same as the number of data"
        )
    features = {}
    for func in funcs:
        if hasattr(feature, func):
            feat: Callable = getattr(feature, func, **kwargs)
            features[func] = np.array(feat(frames, sample_rate, **kwargs))
        else:
            raise ValueError(f"{func} is not a valid feature function")
    return features
