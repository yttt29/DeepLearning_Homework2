"""用已知答案验证报告数据导出，避免混淆矩阵方向或样本索引错误。"""
from pathlib import Path
import csv
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import export_predictions


class EvaluationTests(unittest.TestCase):
    def test_known_predictions_and_error_indices(self):
        logits = torch.full((3, 10), -3.0)
        logits[0, 0] = 3; logits[1, 2] = 4; logits[2, 2] = 3
        labels = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'evaluation'
            result = export_predictions(torch.nn.Identity(), loader, [101, 202, 303],
                                        torch.device('cpu'), out, 'synthetic', list(map(str, range(10))))
            self.assertAlmostEqual(result['accuracy'], 2/3)
            self.assertEqual(result['errors'], 1)
            with np.load(out / 'predictions.npz') as data:
                np.testing.assert_array_equal(data['indices'], [101, 202, 303])
                np.testing.assert_array_equal(data['predictions'], [0, 2, 2])
            with (out / 'confusion_matrix.csv').open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(int(rows[2][3]), 1)  # 真实类别 1，预测类别 2。
            with (out / 'errors.csv').open() as f:
                errors = list(csv.DictReader(f))
            self.assertEqual(errors[0]['original_index'], '202')
            with (out / 'per_class.csv').open() as f:
                recalls = list(csv.DictReader(f))
            self.assertEqual(float(recalls[1]['recall']), 0)
            self.assertEqual(float(recalls[2]['recall']), 1)


if __name__ == '__main__':
    unittest.main()
