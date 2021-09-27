"""
Audio preprocessing utilities for Sotaque Brasileiro project.
"""
from functools import partial
from typing import List, Tuple, Union
from difflib import SequenceMatcher

import webrtcvad
import numpy as np
from p_tqdm import p_map, t_map
from scipy import signal
import speech_recognition as sr
from librosa.effects import preemphasis
from python_speech_features import sigproc

from sotaque_brasileiro.constants import constants
#
# Utilities
#


def speech_to_text(
    audio_file: str,
    *args,
    **kwargs,
) -> str:
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


def str_similarity(
    str_a: str,
    str_b: str
) -> float:
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
    *,
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

#
# Functions for full audio array only
#


def filter_speech(
    audio_array: np.ndarray,
    sample_rate: int,
    *,
    frame_duration_ms: int = 20,
    frame_step_ms: int = 10,
    aggressiveness: int = 1
) -> Tuple[int, List[np.ndarray]]:
    """
    Filters the speech from wav audio.
    :param audio_array: array of the audio
    :param sample_rate: sample rate
    :param frame_duration_ms: duration of each frame in ms
    :param aggressiveness: aggressiveness level (from 0 to 3)
    :return: sample_rate, list of frames
    """
    frames = get_frames(
        audio_array, sample_rate, frame_duration_ms=frame_duration_ms, frame_step_ms=frame_step_ms)
    filtered_frames = []
    for frame in frames:
        if is_speech(frame, sample_rate, aggressiveness):
            filtered_frames.append(frame)
    return filtered_frames


def get_frames(
    audio_array: np.ndarray,
    sample_rate: int,
    *,
    frame_duration_ms: int = 20,
    frame_step_ms: int = 10,
    winfunc=lambda x: np.ones((x,))
) -> np.ndarray:
    """
    Returns the frames of wav audio.
    :param audio_array: array of the audio
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
        audio_array, frame_len=frame_len, frame_step=frame_step, winfunc=winfunc)
    return frames

#
# Functions for audio frames OR full audio arrays
#


def resample(
    audio_array_or_frame: Union[np.ndarray, List[np.ndarray]],
    sample_rate: int,
    new_sample_rate: int,
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Resamples an array.
    :param arr: array to be resampled
    :param sample_rate: sample rate
    :param new_sample_rate: new sample rate
    :return: the resampled array
    """
    if sample_rate == new_sample_rate:
        return audio_array_or_frame
    elif isinstance(audio_array_or_frame, list) and len(audio_array_or_frame) > 0 and isinstance(audio_array_or_frame[0], np.ndarray):
        return t_map(
            partial(signal.resample, num=int(
                len(audio_array_or_frame) * new_sample_rate / sample_rate)),
            audio_array_or_frame,
            desc="Resampling..."
        )
    return signal.resample(audio_array_or_frame, int(len(audio_array_or_frame) * new_sample_rate / sample_rate))


def stereo_to_mono(
    audio_array_or_frame: Union[np.ndarray, List[np.ndarray]],
    *args,
    **kwargs
) -> np.ndarray:
    """
    Converts a stereo array to mono.
    :param arr: array to be converted
    :return: the converted array
    """
    if len(audio_array_or_frame.shape) == 1:
        return audio_array_or_frame
    elif isinstance(audio_array_or_frame, list) and len(audio_array_or_frame) > 0 and isinstance(audio_array_or_frame[0], np.ndarray):
        return t_map(partial(np.mean, axis=1), audio_array_or_frame, desc="Converting to mono...")
    return np.mean(audio_array_or_frame, axis=1)


def apply_filter(
    audio_array_or_frame: Union[np.ndarray, List[np.ndarray]],
    b: np.ndarray,
    a: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:  # pylint: disable=invalid-name
    """
    Applies a filter to an array.
    :param audio_array_or_frame: array to be filtered
    :param b: coefficients of the filter
    :param a: coefficients of the filter
    :return: the filtered array
    """
    if isinstance(audio_array_or_frame, list) and len(audio_array_or_frame) > 0 and isinstance(audio_array_or_frame[0], np.ndarray):
        return t_map(
            partial(signal.lfilter, b, a),
            audio_array_or_frame,
            desc="Applying filter..."
        )
    return signal.filtfilt(b, a, audio_array_or_frame)


def remove_dc_level(
    audio_array_or_frame: Union[np.ndarray, List[np.ndarray]],
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Removes the DC level from an array.
    :param audio_array_or_frame: array to be filtered
    :return: the filtered array
    """
    def remove_mean(arr):
        return arr - np.mean(arr)
    if isinstance(audio_array_or_frame, list) and len(audio_array_or_frame) > 0 and isinstance(audio_array_or_frame[0], np.ndarray):
        return t_map(remove_mean, audio_array_or_frame, desc="Removing DC level...")
    return remove_mean(audio_array_or_frame)


def convert_to_16bit(
    audio_array_or_frame: Union[np.ndarray, List[np.ndarray]],
    *args,
    **kwargs,
) -> np.ndarray:
    """
    Converts an array to 16 bits.
    :param audio_array_or_frame: array to be converted
    :return: the converted array
    """
    def to_16bit(arr):
        return np.int16(arr / np.max(np.abs(arr)) * 32767)
    if isinstance(audio_array_or_frame, list) and len(audio_array_or_frame) > 0 and isinstance(audio_array_or_frame[0], np.ndarray):
        return t_map(
            to_16bit,
            audio_array_or_frame,
            desc="Converting to 16 bit...",
        )
    return to_16bit(audio_array_or_frame)


def apply_preemphasis(
    audio_array_or_frame: Union[np.ndarray, List[np.ndarray]],
    *,
    coef: float = constants.PREEMPHASIS_COEFFICIENT.value
) -> np.ndarray:
    """
    Applies a preemphasis filter to an array.
    :param audio_array_or_frame: array to be filtered
    :param coef: preemphasis coefficient
    :return: the filtered array
    """
    if isinstance(audio_array_or_frame, list) and len(audio_array_or_frame) > 0 and isinstance(audio_array_or_frame[0], np.ndarray):
        return t_map(
            partial(preemphasis, coef=coef),
            audio_array_or_frame,
            desc="Applying preemphasis..."
        )
    return preemphasis(audio_array_or_frame, coef=coef)

#
# Functions for audio frames only
#


def is_speech(
    frame: np.ndarray,
    sample_rate: int,
    *args,
    aggressiveness: int = 1
) -> bool:
    """
    Determines if a frame is speech.
    :param frame: frame to be checked
    :param aggressiveness: aggressiveness level (from 0 to 3)
    :return: True if the frame is speech, False otherwise
    """
    try:
        vad = webrtcvad.Vad(aggressiveness)
        ans = vad.is_speech(frame, sample_rate)
    except:
        print("Error while processing frame, skipping...")
        return constants.DEFAULT_SPEECH_FLAG.value
    return ans
