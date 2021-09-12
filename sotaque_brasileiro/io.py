"""
IO utils for the Sotaque Brasileiro project.
"""
from typing import Iterable
from os import environ
from pathlib import Path

import requests
import numpy as np
from minio import Minio
from scipy.io import wavfile
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from pydub import AudioSegment

from sotaque_brasileiro.constants import constants

__all__ = [
    "load_envs_from_file",
    "load_wav_file",
    "save_envs_to_file",
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
    # pylint: disable=import-outside-toplevel
    try:
        return environ[key]
    except KeyError as exc:
        env_file = Path(constants.ENV_FILE_DEFAULT_PATH.value)
        if env_file.exists() and env_file.is_file():
            try:
                load_envs_from_file()
                return environ[key]
            except KeyError as exc:
                raise KeyError(
                    f"{key} is not set and the default env file was not found."
                ) from exc
        else:
            raise KeyError(
                f"{key} is not set and the default env file was not found."
            ) from exc


def load_wav_file(file_path: str):
    """
    Load WAV data from file.
    """
    rate, data = wavfile.read(file_path)
    return rate, data


def fetch_paginated_data(url):
    """
    Fetch paginated data from a url and return a list of all the data.
    """
    data = []
    while url:
        response = requests.get(url)
        response_json = response.json()
        data.extend(response_json["results"])
        url = response_json["next"]
    return data


def load_envs_from_file(file_path=constants.ENV_FILE_DEFAULT_PATH.value):
    """
    Load environment variables from a file.
    """
    #pylint: disable=unspecified-encoding
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            environ[key] = value


def save_envs_to_file(file_path=constants.ENV_FILE_DEFAULT_PATH.value):
    """
    Save environment variables to a file.
    """
    #pylint: disable=unspecified-encoding
    with open(file_path, "w") as file:
        for key, value in environ.items():
            if key in constants.ENV_FILE_ALLOWED_KEYS.value:
                file.write("{}={}\n".format(key, value))


def download_file(bucket_name: str, object_name: str, file_path: str):
    """
    Download a file from a bucket.
    """
    # pylint: disable=import-outside-toplevel
    from sotaque_brasileiro.utils import safe_getenv

    minio_client = Minio(
        safe_getenv(constants.MINIO_ENDPOINT.value),
        access_key=safe_getenv(constants.MINIO_ACCESS_KEY.value),
        secret_key=safe_getenv(constants.MINIO_SECRET_KEY.value),
    )
    minio_client.fget_object(bucket_name, object_name, file_path)


def download_multiple(
    bucket_name: str,
    object_names: Iterable[str],
    file_paths: Iterable[str],
    show_progress: bool = constants.SHOW_PROGRESS_DEFAULT.value,
):
    """
    Download multiple files from a bucket.
    """
    # pylint: disable=import-outside-toplevel
    from sotaque_brasileiro.utils import safe_getenv

    object_names = list(object_names)
    file_paths = list(file_paths)
    minio_client = Minio(
        safe_getenv(constants.MINIO_ENDPOINT.value),
        access_key=safe_getenv(constants.MINIO_ACCESS_KEY.value),
        secret_key=safe_getenv(constants.MINIO_SECRET_KEY.value),
    )
    if show_progress:
        if tqdm is None:
            raise ImportError(
                """tqdm must be installed to show progress.
                Either install tqdm or run with show_progress=False"""
            )
        for object_name, file_path in tqdm(
            zip(object_names, file_paths),
            desc="Downloading audio files...",
            total=len(object_names),
        ):
            minio_client.fget_object(bucket_name, object_name, file_path)
            tqdm.write(object_name)
    else:
        for object_name, file_path in zip(object_names, file_paths):
            minio_client.fget_object(bucket_name, object_name, file_path)


def webm_to_wav(webm_file: str):
    """
    Converts a webm file to a wav file.
    :param webm_file: path to the webm file
    :return: path to the wav file
    """
    wav_file = webm_file.replace(".webm", ".wav")
    wav = AudioSegment.from_file(webm_file)
    wav.export(wav_file, format="wav")
    return wav_file


def save_frames_to_wav_file(frames: np.ndarray, sample_rate: int, file_path: str):
    """
    Save a list of frames to a WAV file.
    """
    wavfile.write(file_path, sample_rate, np.hstack(frames))
