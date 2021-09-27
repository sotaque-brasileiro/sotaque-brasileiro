"""
Constants for the Sotaque Brasileiro project.
"""

from enum import Enum
from pathlib import Path

__all__ = ["constants"]


class constants(Enum):  # pylint: disable=invalid-name
    """
    Enum with all constants used in the project.
    """

    # General
    API_BASE_URL = "https://api-sotaque.gabriel-milan.com/"
    API_RECORDS_ENDPOINT = API_BASE_URL + "records/"

    # Defaults
    SHOW_PROGRESS_DEFAULT = False
    TIMEZONE_DEFAULT = "America/Sao_Paulo"
    DATASET_SAVE_DEFAULT_PATH = "./sotaque-brasileiro-data"
    DEFAULT_SPEECH_FLAG = True
    STEREO_TO_MONO_DEFAULT = False

    # Environment
    ENV_FILE_DEFAULT_PATH = Path.home() / ".sotaque_brasileiro.env"
    ENV_FILE_ALLOWED_KEYS = [
        "GCS_BUCKET_NAME",
        "GOOGLE_CLOUD_CREDENTIALS",
    ]

    # Preprocessing
    PREEMPHASIS_COEFFICIENT = 0.95

    # Visualization
    BRAZIL_CENTER_DEFAULT = dict(lat=-15.822, lon=-47.611)

    # Google Cloud env names
    GCS_BUCKET_NAME = "GCS_BUCKET_NAME"
    GOOGLE_CLOUD_CREDENTIALS = "GOOGLE_CLOUD_CREDENTIALS"
