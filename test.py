import unittest
import numpy as np

from detector import get_max_similar_part, get_similarity
from utils import open_wav_mono


class TestSum(unittest.TestCase):
    def test_open_stereo_as_mono(self):
        audio, _ = open_wav_mono("test_files/full.wav")
        self.assertEqual(np.ndim(audio), 1)

    def test_reference_part_of_full(self):
        full, _ = open_wav_mono("test_files/full.wav")
        part, _ = open_wav_mono("test_files/reference.wav")
        max_similar_part = get_max_similar_part(full, part)
        similarity = get_similarity(part, max_similar_part)
        self.assertGreaterEqual(similarity, 0.99)

    def test_reference_quieter_than_full(self):
        full, _ = open_wav_mono("test_files/full.wav")
        part, _ = open_wav_mono("test_files/quieter.wav")
        max_similar_part = get_max_similar_part(full, part)
        similarity = get_similarity(part, max_similar_part)
        self.assertGreaterEqual(similarity, 0.99)

    def test_reference_louder_than_full(self):
        full, _ = open_wav_mono("test_files/full.wav")
        part, _ = open_wav_mono("test_files/louder.wav")
        max_similar_part = get_max_similar_part(full, part)
        similarity = get_similarity(part, max_similar_part)
        self.assertGreaterEqual(similarity, 0.99)

    def test_reference_part_of_full_with_bg(self):
        full, _ = open_wav_mono("test_files/full_bm.wav")
        part, _ = open_wav_mono("test_files/reference.wav")
        max_similar_part = get_max_similar_part(full, part)
        similarity = get_similarity(part, max_similar_part)
        self.assertGreaterEqual(similarity, 0.99)

    def test_reference_not_in_full(self):
        full, _ = open_wav_mono("test_files/no_reference.wav")
        part, _ = open_wav_mono("test_files/reference.wav")
        max_similar_part = get_max_similar_part(full, part)
        similarity = get_similarity(part, max_similar_part)
        self.assertLessEqual(similarity, 0.5)

    def test_reference_part_of_full_with_loud_bg(self):
        full, _ = open_wav_mono("test_files/full_loud.wav")
        part, _ = open_wav_mono("test_files/reference.wav")
        max_similar_part = get_max_similar_part(full, part)
        similarity = get_similarity(part, max_similar_part)
        self.assertGreaterEqual(similarity, 0.5)


if __name__ == '__main__':
    unittest.main()
