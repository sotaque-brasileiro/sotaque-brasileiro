import requests
from typing import Iterable
from os import environ

from minio import Minio
from scipy.io import wavfile

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from sotaque_brasileiro.constants import constants

__all__ = [
    "load_envs_from_file",
    "load_wav_file",
    "save_envs_to_file",
]


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
    with open(file_path, "w") as file:
        for key, value in environ.items():
            if key in constants.ENV_FILE_ALLOWED_KEYS.value:
                file.write("{}={}\n".format(key, value))


def download_file(bucket_name: str, object_name: str, file_path: str):
    """
    Download a file from a bucket.
    """
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
                "tqdm must be installed to show progress. Either install tqdm or run this command with show_progress=False"
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
