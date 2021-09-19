"""
Feature extraction for Sotaque Brasileiro.
"""

import numpy as np
import librosa.core
import librosa.feature


def chroma_stft(frames, sample_rate, *, kwargs={}):
    """
    Extracts chroma_stft features from the given frames.
    """
    l = []
    for frame in frames:
        l.append(
            np.mean(
                librosa.feature.chroma_stft(
                    y=frame,
                    sr=sample_rate,
                    **kwargs
                ).T, axis=0
            )
        )
    return np.array(l)


def chroma_cqt(frames, sample_rate, *, kwargs={}):
    """
    Extracts chroma_cqt features from the given frames.
    """
    l = []
    for frame in frames:
        l.append(
            np.mean(
                librosa.feature.chroma_cqt(
                    y=frame,
                    sr=sample_rate,
                    **kwargs
                ).T, axis=0
            )
        )
    return np.array(l)


def chroma_cens(frames, sample_rate, *, kwargs={}):
    """
    Extracts chroma_cens features from the given frames.
    """
    l = []
    for frame in frames:
        l.append(
            np.mean(
                librosa.feature.chroma_cens(
                    y=frame,
                    sr=sample_rate,
                    **kwargs
                ).T, axis=0
            )
        )
    return np.array(l)


def log_energy(frames, sample_rate, *, kwargs={}):
    """
    Extracts log-mel energies from the given frames.
    """
    l = []
    for frame in frames:
        energy = librosa.feature.melspectrogram(
            y=frame, sr=sample_rate, power=1, **kwargs)
        l.append(
            np.mean(
                librosa.core.amplitude_to_db(
                    energy
                ).T, axis=0
            )
        )
    return np.array(l)


def mfcc(
    frames: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
    dct_type: int = 2,
    lifter: int = 0,
    *,
    kwargs={},
):
    """
    Extracts MFCC features from the given frames.
    """
    l = []
    for frame in frames:
        l.append(
            np.mean(
                librosa.feature.mfcc(
                    y=frame,
                    sr=sample_rate,
                    n_mfcc=n_mfcc,
                    dct_type=dct_type,
                    lifter=lifter,
                    **kwargs
                ).T, axis=0
            )
        )
    return np.array(l)


def poly_features(frames, sample_rate, *, kwargs={}):
    """
    Extracts poly features from the given frames.
    """
    l = []
    for frame in frames:
        l.append(
            np.mean(
                librosa.feature.poly_features(
                    y=frame,
                    sr=sample_rate,
                    **kwargs
                ).T, axis=0
            )
        )
    return np.array(l)


def spectral_contrast(
    frames,
    sample_rate,
    *,
    kwargs={},
):
    """
    Extracts spectral contrast from the given frames.
    """
    l = []
    for frame in frames:
        l.append(
            np.mean(
                librosa.feature.spectral_contrast(
                    y=frame,
                    sr=sample_rate,
                    **kwargs
                ).T, axis=0
            )
        )
    return np.array(l)


def sdc(
    frames: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 20,
    dct_type: int = 2,
    lifter: int = 0,
    *,
    kwargs={},
):
    """
    Extracts SDC features from the given frames.
    """
    mfcc_features = mfcc(frames, sample_rate, n_mfcc,
                         dct_type, lifter, **kwargs)
    mfcc_delta = librosa.feature.delta(mfcc_features)
    mfcc_delta2 = librosa.feature.delta(mfcc_features, order=2)
    return np.concatenate((mfcc_delta, mfcc_delta2), axis=1)


def zero_crossing_rate(
    frames,
    sample_rate,
    *,
    kwargs={},
):
    """
    Extracts zero crossing rate from the given frames.
    """
    l = []
    for frame in frames:
        l.append(librosa.feature.zero_crossing_rate(
            y=frame, frame_length=len(frame))[0][0], **kwargs)
    return np.array(l)
