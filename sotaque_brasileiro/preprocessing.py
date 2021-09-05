from typing import List, Tuple
from difflib import SequenceMatcher

import webrtcvad
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr

from sotaque_brasileiro.io import load_wav_file


def filter_speech(
    audio_file: str, frame_duration_ms: int, aggressiveness: int = 1
) -> Tuple[int, List[np.ndarray]]:
    """
    Filters the speech from an audio file.
    :param audio_file: path to the audio file
    :param frame_duration_ms: duration of each frame in ms
    :param aggressiveness: aggressiveness level (from 0 to 3)
    :return: sample_rate, list of frames
    """
    sr, frames = get_frames(audio_file, frame_duration_ms)
    filtered_frames = []
    for frame in frames:
        if is_speech(frame, sr, aggressiveness):
            filtered_frames.append(frame)
    return sr, filtered_frames


def get_frames(wav_file: str, frame_duration_ms: int) -> Tuple[int, List[np.ndarray]]:
    """
    Returns the frames of a wav file.
    :param wav_file: path to the wav file
    :param frame_duration_ms: duration of each frame in ms
    :return: sample_rate, list of frames
    """
    frames = []
    sr, data = load_wav_file(wav_file)
    for i in range(0, len(data), int(sr * frame_duration_ms / 1000)):
        frames.append(data[i : i + int(sr * frame_duration_ms / 1000)])
    return sr, frames


def is_speech(frame: np.ndarray, sample_rate: int, aggressiveness: int = 1) -> bool:
    """
    Determines if a frame is speech.
    :param frame: frame to be checked
    :param aggressiveness: aggressiveness level (from 0 to 3)
    :return: True if the frame is speech, False otherwise
    """
    vad = webrtcvad.Vad(aggressiveness)
    return vad.is_speech(frame, sample_rate)


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


def speech_to_text(audio_file: str):
    """
    Converts an audio file to text using Google Speech Recognition.
    :param audio_file: path to the audio file
    :return: text
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language="pt-BR")
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise sr.RequestError("No internet connection")


def str_similarity(a: str, b: str):
    """
    Returns the similarity between two strings.
    :param a: first string
    :param b: second string
    :return: similarity
    """
    return SequenceMatcher(None, a, b).ratio()
