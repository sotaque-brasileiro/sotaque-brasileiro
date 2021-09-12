"""
Audio preprocessing utilities for Sotaque Brasileiro project.
"""
from typing import List, Tuple
from difflib import SequenceMatcher

import webrtcvad
import numpy as np
from scipy import signal
import speech_recognition as sr
from librosa.effects import preemphasis
from python_speech_features import sigproc

from sotaque_brasileiro.constants import constants


def filter_speech(
    data: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = 20,
    frame_step_ms: int = 10,
    aggressiveness: int = 1
) -> Tuple[int, List[np.ndarray]]:
    """
    Filters the speech from wav audio.
    :param data: array of the audio
    :param sample_rate: sample rate
    :param frame_duration_ms: duration of each frame in ms
    :param aggressiveness: aggressiveness level (from 0 to 3)
    :return: sample_rate, list of frames
    """
    frames = get_frames(
        data, sample_rate, frame_duration_ms=frame_duration_ms, frame_step_ms=frame_step_ms)
    filtered_frames = []
    for frame in frames:
        if is_speech(frame, sample_rate, aggressiveness):
            filtered_frames.append(frame)
    return sr, filtered_frames


def get_frames(
    data: np.ndarray,
    sample_rate: int,
    frame_duration_ms: int = 20,
    frame_step_ms: int = 10,
    winfunc=lambda x: np.ones((x,))
) -> np.ndarray:
    """
    Returns the frames of wav audio.
    :param data: array of the audio
    :param sample_rate: sample rate
    :param frame_duration_ms: duration of each frame in ms, default 20ms
    :param frame_step_ms: number of samples after the start of the previous
        frame that the next frame should begin, default 10ms
    :param winfunc: window function, default no window
    :return: the frames
    """
    frame_len = int(frame_duration_ms / 1000 * sample_rate)
    frame_step = int(frame_step_ms / 1000 * sample_rate)
    frames = sigproc.framesig(
        data, frame_len=frame_len, frame_step=frame_step, winfunc=winfunc)
    return frames


def is_speech(frame: np.ndarray, sample_rate: int, aggressiveness: int = 1) -> bool:
    """
    Determines if a frame is speech.
    :param frame: frame to be checked
    :param aggressiveness: aggressiveness level (from 0 to 3)
    :return: True if the frame is speech, False otherwise
    """
    vad = webrtcvad.Vad(aggressiveness)
    return vad.is_speech(frame, sample_rate)


def speech_to_text(audio_file: str):
    """
    Converts an audio file to text using Google Speech Recognition.
    :param audio_file: path to the audio file
    :return: text
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="pt-BR")
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:
        raise exc


def str_similarity(str_a: str, str_b: str):
    """
    Returns the similarity between two strings.
    :param a: first string
    :param b: second string
    :return: similarity
    """
    return SequenceMatcher(None, str_a, str_b).ratio()


def get_filter(
    sample_rate: int,
    frequency: int,
    btype: str,
    order: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a filter for the given parameters.
    :param sample_rate: sample rate
    :param frequency: frequency
    :param btype: type of filter
    :param order: order of the filter
    :return: the coefficients
    """
    nyq = 0.5 * sample_rate
    high = frequency / nyq
    #pylint: disable=invalid-name
    b, a = signal.butter(order, high, btype=btype)
    return b, a


def apply_filter(arr: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Applies a filter to an array.
    :param arr: array to be filtered
    :param b: coefficients of the filter
    :param a: coefficients of the filter
    :return: the filtered array
    """
    return signal.filtfilt(b, a, arr)


def remove_dc_level(arr: np.ndarray) -> np.ndarray:
    """
    Removes the DC level from an array.
    :param arr: array to be filtered
    :return: the filtered array
    """
    return arr - np.mean(arr)


def convert_to_16bit(arr: np.ndarray) -> np.ndarray:
    """
    Converts an array to 16 bits.
    :param arr: array to be converted
    :return: the converted array
    """
    return np.int16(arr / np.max(np.abs(arr)) * 32767)


def apply_preemphasis(
    arr: np.ndarray,
    coef: float = constants.PREEMPHASIS_COEFFICIENT.value
) -> np.ndarray:
    """
    Applies a preemphasis filter to an array.
    :param arr: array to be filtered
    :param coef: preemphasis coefficient
    :return: the filtered array
    """
    return preemphasis(arr, coef=coef)
