import unittest

import torch

from models import DeepConvNet, EEGNet82, GraphEEGNet
from models.eegnet import AdaptiveTemporalAvgPool2d


MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class ModelShapeTests(unittest.TestCase):
    def _assert_logits(self, model, time_points):
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 1, 24, time_points))
        self.assertEqual(tuple(output.shape), (2, 26))
        self.assertTrue(torch.isfinite(output).all())

    def test_deepconvnet_window_shape(self):
        self._assert_logits(
            DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5), 501
        )

    def test_eegnet_k15_and_k25_shapes(self):
        for kernel in (15, 25):
            with self.subTest(kernel=kernel):
                self._assert_logits(
                    EEGNet82(
                        24,
                        26,
                        input_time_points=801,
                        temporal_kernel_length=kernel,
                        dropout_rate=0.3,
                    ),
                    801,
                )

    def test_graph_eeg_shape(self):
        self._assert_logits(GraphEEGNet(), 501)

    def test_adaptive_temporal_pool_matches_pytorch(self):
        for input_width, output_width in ((126, 16), (201, 16), (19, 8)):
            with self.subTest(input_width=input_width, output_width=output_width):
                inputs = torch.randn(2, 3, 2, input_width)
                expected = torch.nn.functional.adaptive_avg_pool2d(
                    inputs, (1, output_width)
                )
                actual = AdaptiveTemporalAvgPool2d(output_width)(inputs)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(MPS_AVAILABLE, "Apple Metal is unavailable")
    def test_eegnet_nondivisible_pooling_on_mps(self):
        model = EEGNet82(
            24,
            26,
            input_time_points=801,
            temporal_kernel_length=25,
            dropout_rate=0.3,
        ).to("mps")
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 1, 24, 801, device="mps"))
        self.assertEqual(tuple(output.shape), (2, 26))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
