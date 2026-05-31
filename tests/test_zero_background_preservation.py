import unittest

import numpy as np

from image_pipeline import ImageProcessingPipeline, PipelineConfig


class ZeroBackgroundPreservationTest(unittest.TestCase):
    def test_zero_background_mask_is_reapplied_without_mutating_input(self):
        pipeline = ImageProcessingPipeline(
            PipelineConfig(preserve_zero_background=True)
        )
        reference = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )
        enhanced = np.array(
            [
                [100, 200, 300],
                [400, 500, 600],
            ],
            dtype=np.uint16,
        )

        mask = pipeline._reference_background_mask(reference)
        masked = pipeline._zero_background(enhanced, mask)

        np.testing.assert_array_equal(enhanced[reference == 0], [100, 200, 400])
        self.assertTrue(np.all(masked[reference == 0] == 0))
        np.testing.assert_array_equal(masked[reference > 0], enhanced[reference > 0])

    def test_zero_background_preservation_can_be_disabled(self):
        pipeline = ImageProcessingPipeline(
            PipelineConfig(preserve_zero_background=False)
        )
        reference = np.array([[0.0, 1.0]], dtype=np.float32)

        self.assertIsNone(pipeline._reference_background_mask(reference))

    def test_background_threshold_can_include_near_zero_pixels(self):
        pipeline = ImageProcessingPipeline(
            PipelineConfig(
                preserve_zero_background=True,
                background_zero_threshold=0.1,
            )
        )
        reference = np.array([[0.0, 0.05, 0.2]], dtype=np.float32)

        mask = pipeline._reference_background_mask(reference)

        np.testing.assert_array_equal(mask, [[True, True, False]])


if __name__ == "__main__":
    unittest.main()
