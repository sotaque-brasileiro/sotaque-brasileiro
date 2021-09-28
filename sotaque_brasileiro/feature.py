"""
Feature extraction for Sotaque Brasileiro.
"""

from typing import List

import numpy as np
import librosa.core
import librosa.feature
from p_tqdm import p_map


def chroma_stft(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts chroma_stft features from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return np.mean(
            librosa.feature.chroma_stft(
                y=frame,
                sr=sample_rate,
                **kwargs
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting Chroma STFT")


def chroma_cqt(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts chroma_cqt features from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return np.mean(
            librosa.feature.chroma_cqt(
                y=frame,
                sr=sample_rate,
                **kwargs
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting Chroma CQT")


def chroma_cens(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts chroma_cens features from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return np.mean(
            librosa.feature.chroma_cens(
                y=frame,
                sr=sample_rate,
                **kwargs
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting Chroma CENS")


def log_energy(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts log-mel energies from the given frames.
    """
    def proc_frame(frame, sample_rate):
        energy = librosa.feature.melspectrogram(
            y=frame, sr=sample_rate, power=1, **kwargs)
        return np.mean(
            librosa.core.amplitude_to_db(
                energy
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting log-energy")


def mfcc(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    n_mfcc: int = 20,
    dct_type: int = 2,
    lifter: int = 0,
    kwargs={},
) -> List[np.ndarray]:
    """
    Extracts MFCC features from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return np.mean(
            librosa.feature.mfcc(
                y=frame,
                sr=sample_rate,
                n_mfcc=n_mfcc,
                dct_type=dct_type,
                lifter=lifter,
                **kwargs
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting MFCC")


def poly_features(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts poly features from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return np.mean(
            librosa.feature.poly_features(
                y=frame,
                sr=sample_rate,
                **kwargs
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting polynomial features")


def spectral_contrast(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts spectral contrast from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return np.mean(
            librosa.feature.spectral_contrast(
                y=frame,
                sr=sample_rate,
                **kwargs
            ).T, axis=0
        )
    return p_map(proc_frame, frames, sample_rate, desc="Extracting spectral constrast")


def sdc(
    frames: List[np.ndarray],
    sample_rate: List[int],
    n_mfcc: int = 20,
    dct_type: int = 2,
    lifter: int = 0,
    *,
    kwargs={},
) -> List[np.ndarray]:
    """
    Extracts SDC features from the given frames.
    """
    mfcc_features = mfcc(frames, sample_rate, n_mfcc=n_mfcc,
                         dct_type=dct_type, lifter=lifter, kwargs=kwargs)

    def proc_frame(mfcc_feature, sample_rate):
        mfcc_delta = librosa.feature.delta(mfcc_feature)
        mfcc_delta2 = librosa.feature.delta(mfcc_delta)
        return np.concatenate((mfcc_delta, mfcc_delta2))
    return p_map(proc_frame, mfcc_features, sample_rate, desc="Extracting SDC")


def zero_crossing_rate(
    frames: List[np.ndarray],
    sample_rate: List[int],
    *,
    kwargs: dict = {}
) -> List[np.ndarray]:
    """
    Extracts zero crossing rate from the given frames.
    """
    def proc_frame(frame, sample_rate):
        return librosa.feature.zero_crossing_rate(
            y=frame,
            frame_length=len(frame),
            **kwargs,)[0][0]
    return p_map(proc_frame, frames, sample_rate, desc="Extracting zero-crossing rate")
