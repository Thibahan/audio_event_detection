from scipy import signal
import numpy as np

from utils import open_wav_mono


def get_max_similar_part(
    full_audio: np.ndarray, reference_audio: np.ndarray
) -> np.ndarray:
    """This function will return the most similar part of the full audio file
    to the reference audio file.

    Args:
        full_audio (np.ndarray): Full audio file to find the reference in.
        Must be larger then reference audio. Must be 1D.
        reference_audio (np.ndarray): Reference to search in full audio file.
        Must be 1D

    Returns:
        np.ndarray: Returns the most similar part from the full audio.
    """
    if np.ndim(full_audio) != 1 or np.ndim(full_audio) != 1:
        print("ValueError: Only works with 1d arrays as input")
        raise ValueError

    if len(full_audio) <= len(reference_audio):
        print("ValueError: Full audio should be larger than refrence audio")
        raise ValueError

    len_reference_audio = len(reference_audio)
    conv_correlation = signal.correlate(
        full_audio, reference_audio, mode="valid", method="fft"
    )
    peak = np.argmax(conv_correlation)
    max_similar_part = full_audio[peak: peak + len_reference_audio]
    return max_similar_part


def get_similarity(reference_audio: np.ndarray, max_similar_part: np.ndarray) -> float:
    """This function will calculate the pearson product-moment correlation
    coefficients of two given files. Both files must have the same size.

    Args:
        reference_audio (np.ndarray): Audiofile one. Must be 1D.
        max_similar_part (np.ndarray): Audio file two. Must be 1D.

    Returns:
        float: The pearson product-moment correlation coefficient.
    """
    if len(reference_audio) != len(max_similar_part):
        print("ValueError: Input arrays must have the same size")
        raise ValueError

    if np.ndim(reference_audio) != 1 or np.ndim(max_similar_part) != 1:
        print("ValueError: Only works with 1d arrays as input")
        raise ValueError

    corr_coef = np.corrcoef(reference_audio, max_similar_part)
    cc_xy = corr_coef[0][1]
    cc_yx = corr_coef[1][0]
    corr_coef = (np.sqrt(cc_xy**2) + np.sqrt(cc_yx**2)) / 2
    return corr_coef


if __name__ == "__main__":
    full, _ = open_wav_mono("test_files/full.wav")
    part, _ = open_wav_mono("test_files/reference_mono.wav")
    max_similar_part = get_max_similar_part(full, part)
    similarity = get_similarity(part, max_similar_part)
    print("Reference found with an similyrity of {:.2f}".format(similarity))
