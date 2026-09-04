import unittest

import torch

from models import DeepConvNet, EEGNet82, GraphEEGNet


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


if __name__ == "__main__":
    unittest.main()
