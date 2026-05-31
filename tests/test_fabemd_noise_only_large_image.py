import contextlib
import io
import unittest

import numpy as np

from fabemd import FABEMD


class FABEMDNoiseOnlyLargeImageTest(unittest.TestCase):
    def test_decomposes_noise_only_2959_by_3685_image(self):
        rng = np.random.default_rng(29593685)
        image = rng.normal(loc=0.0, scale=1.0, size=(2959, 3685)).astype(np.float32)

        fabemd = FABEMD(
            max_sift_iterations=1,
            max_bimfs=1,
            min_extrema=1,
            initial_window_size=3,
            window_size_cap=9,
            extrema_window=3,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            bimfs = fabemd.decompose(image)

        self.assertEqual(1, len(bimfs))
        self.assertEqual(image.shape, bimfs[0].shape)
        self.assertTrue(np.isfinite(bimfs[0]).all())
        self.assertEqual([(5, 5)], fabemd.window_sizes_)

        residual = image.astype(np.float64) - sum(bimfs)
        self.assertEqual(image.shape, residual.shape)
        self.assertTrue(np.isfinite(residual).all())


if __name__ == "__main__":
    unittest.main()
