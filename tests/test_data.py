"""检查固定划分、训练顺序与批次格式。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import create_data_loaders


class DataLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val = create_data_loaders()

    def test_saved_split_and_preprocessing(self):
        self.assertEqual(len(self.train.dataset), 50000)
        self.assertEqual(len(self.val.dataset), 10000)
        indices = self.train.dataset.indices + self.val.dataset.indices
        self.assertEqual(sorted(indices), list(range(60000)))
        images, labels = next(iter(self.val))
        self.assertEqual(tuple(images.shape), (128, 784))
        self.assertEqual(tuple(labels.shape), (128,))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.int64)
        self.assertTrue(((images >= 0) & (images <= 1)).all())
        self.assertTrue(((labels >= 0) & (labels < 10)).all())
        # 第一张验证图片必须对应保存索引中的第一个样本。
        idx = self.val.dataset.indices[0]
        torch.testing.assert_close(images[0], self.val.dataset.dataset.tensors[0][idx])

    def test_shuffle_reproducibility_and_epoch_change(self):
        generator = self.train.generator
        state = generator.get_state()
        first = list(self.train.sampler)
        second = list(self.train.sampler)
        self.assertNotEqual(first, second)
        self.assertEqual(sorted(first), list(range(50000)))
        generator.set_state(state)
        torch.rand(100)  # 模拟模型初始化对全局随机数的消耗。
        self.assertEqual(first, list(self.train.sampler))
        generator.set_state(state)

    def test_no_samples_dropped(self):
        for loader, expected_count, expected_last in [(self.train, 50000, 80), (self.val, 10000, 16)]:
            count = 0
            for images, labels in loader:
                count += len(labels)
                last = len(labels)
            self.assertEqual(count, expected_count)
            self.assertEqual(last, expected_last)
        self.assertEqual(list(self.val.sampler), list(range(10000)))


if __name__ == '__main__':
    unittest.main()
