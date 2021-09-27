"""
IO utils for the Sotaque Brasileiro project.
"""
import json
import base64
from typing import Iterable
from os import environ
from pathlib import Path

import requests
import numpy as np
from scipy.io import wavfile
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from p_tqdm import p_map
from pydub import AudioSegment
from google.oauth2 import service_account
from google.cloud import storage

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


def load_multiple_wav(file_paths: Iterable[str]):
    """
    Load multiple WAV files.
    """
    return p_map(load_wav_file, file_paths, desc="Loading WAV files...")


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


def get_credentials_from_env() -> service_account.Credentials:
    """Gets credentials from env vars"""
    env: str = safe_getenv(constants.GOOGLE_CLOUD_CREDENTIALS.value)
    if env == "":
        raise ValueError(
            f"GOOGLE_CLOUD_CREDENTIALS env var not set!")
    info: dict = json.loads(base64.b64decode(env))
    return service_account.Credentials.from_service_account_info(info)


def get_gcs_client() -> storage.Client:
    """
    Returns a Google Cloud Storage client.
    """
    credentials = get_credentials_from_env()
    return storage.Client(credentials=credentials)


def download_file(bucket_name: str, object_name: str, file_path: str):
    """
    Download a file from GCS.
    """
    parent_dir = Path(file_path).parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    client = get_gcs_client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(object_name)
    blob.download_to_filename(file_path)


def download_multiple(
    bucket_name: str,
    object_names: Iterable[str],
    file_paths: Iterable[str],
    show_progress: bool = constants.SHOW_PROGRESS_DEFAULT.value,
):
    """
    Download multiple files from a bucket.
    """
    object_names = list(object_names)
    file_paths = list(file_paths)
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
            download_file(bucket_name, object_name, file_path)
            tqdm.write(object_name)
    else:
        for object_name, file_path in zip(object_names, file_paths):
            download_file(bucket_name, object_name, file_path)


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
