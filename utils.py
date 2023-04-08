import soundfile as sf
import numpy as np
from typing import Tuple


def open_wav_mono(path: str, mode: str = "mean") -> Tuple[np.ndarray, int]:
    """This function will open a wav file as a single channel file.
    The mode "mean" will return the mean of all channels.
    The mode "first" will return just the first channel.

    Args:
        path (str): Path to audio file.
        mode (str, optional): Mode to convert multi channels.
        Defaults to "mean".

    Returns:
        Tuple[np.ndarray, int]: Mono sound array, Samplerate.
    """
    sound, sr = sf.read(path)
    if np.ndim(sound) != 1:
        sound = list(sound)
        if mode == "mean":
            sound = [x.mean() for x in sound]
        if mode == "first":
            sound = [x[0] for x in sound]
        sound = np.array(sound)
    return sound, sr
