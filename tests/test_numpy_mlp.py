"""验证梯度检查器本身，避免“检查通过”只是检查器失效。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy_mlp as mlp


class NumpyMLPTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.2, 0.5, 0.1], [0.8, 0.3, 0.4]], dtype=np.float64)
        self.y = np.array([0, 1])
        self.params = mlp.initialize_parameters(3, 4, 2)
        # 隐藏单元同时覆盖正、负半轴，并远离零点。
        self.params['b1'][:] = [1.5, -1.5, 1.5, -1.5]

    def test_all_small_network_parameters_and_restoration(self):
        before = {k: v.copy() for k, v in self.params.items()}
        report = mlp.check_gradients(self.x, self.y, self.params, samples_per_tensor=100)
        self.assertTrue(report['passed'])
        self.assertEqual(sum(s['checked'] for s in report['summary'].values()), 26)
        self.assertEqual(sum(s['skipped'] for s in report['summary'].values()), 0)
        for key in before:
            np.testing.assert_array_equal(before[key], self.params[key])

    def test_wrong_gradient_is_rejected(self):
        original_backward = mlp.backward
        def broken_backward(params, cache):
            grads = original_backward(params, cache)
            grads['b2'] += 0.1  # 模拟一个可以运行、但公式写错的实现。
            return grads
        with patch.object(mlp, 'backward', broken_backward):
            report = mlp.check_gradients(self.x, self.y, self.params)
        self.assertFalse(report['passed'])
        self.assertEqual(report['summary']['b2']['failed'], 2)

    def test_stable_loss_with_extreme_logits(self):
        self.params['W2'][:] = 0
        self.params['b2'][:] = [1000, -1000]
        loss, cache = mlp.forward(self.x, self.y, self.params)
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, 1000.0)
        np.testing.assert_allclose(cache['probs'].sum(axis=1), 1)
        for gradient in mlp.backward(self.params, cache).values():
            self.assertTrue(np.isfinite(gradient).all())

    def test_relu_kink_is_reported_as_skipped(self):
        self.params['W1'][:] = 0
        self.params['b1'][:] = 0
        report = mlp.check_gradients(self.x, self.y, self.params)
        self.assertGreater(report['summary']['b1']['skipped'], 0)
        self.assertFalse(report['passed'])  # 整组参数没有有效检查时，不宣称通过。


if __name__ == '__main__':
    unittest.main()
