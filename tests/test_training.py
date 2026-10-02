"""验证受控初始化、线性等价性、训练更新与验证不更新。"""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import build_model, MODEL_NAMES
from train import run_epoch


class TrainingTests(unittest.TestCase):
    def test_models_and_controlled_initialization(self):
        linear = build_model('linear2', hidden_dim=8, seed=42)
        mlp = build_model('mlp', hidden_dim=8, seed=42)
        for key, value in linear.state_dict().items():
            self.assertTrue(torch.equal(value, mlp.state_dict()[key]))
        x = torch.randn(4, 784)
        for name in MODEL_NAMES:
            self.assertEqual(build_model(name)(x).shape, (4, 10))
        # 两个仿射变换可合并：W = W2 W1，b = W2 b1 + b2。
        weight = linear.output.weight @ linear.hidden.weight
        bias = linear.output.weight @ linear.hidden.bias + linear.output.bias
        torch.testing.assert_close(linear(x), x @ weight.T + bias, atol=2e-6, rtol=2e-5)

    def test_validation_uses_sample_weighting_and_preserves_parameters(self):
        model = build_model('softmax')
        torch.manual_seed(10)
        x, y = torch.randn(5, 784), torch.tensor([0, 1, 2, 3, 4])
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        expected = nn.CrossEntropyLoss()(model(x), y).item()
        result = run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device('cpu'))
        self.assertEqual(result['samples'], 5)
        self.assertAlmostEqual(result['loss'], expected, places=6)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device('cpu'), optimizer)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
