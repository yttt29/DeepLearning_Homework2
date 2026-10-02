"""Download official Fashion-MNIST and persist a deterministic 50k/10k split."""
from pathlib import Path
import json
import numpy as np
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
# HTTPS mirror in the dataset authors' official repository.
FashionMNIST.mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/']
train = FashionMNIST(DATA, train=True, download=True)
test = FashionMNIST(DATA, train=False, download=True)
assert train.data.shape == (60000, 28, 28)
assert test.data.shape == (10000, 28, 28)
seed = 42
split_path = DATA / 'split_seed42.npz'
order = np.random.default_rng(seed).permutation(len(train))
expected = {'train': order[:50000], 'validation': order[50000:]}
if split_path.exists():
    with np.load(split_path) as saved:
        for key in expected:
            assert np.array_equal(saved[key], expected[key]), 'Existing split differs; refusing overwrite'
else:
    np.savez_compressed(split_path, **expected)
files = {str(p.relative_to(DATA)): p.stat().st_size for p in DATA.rglob('*') if p.is_file()}
report = {'source': FashionMNIST.mirrors[0], 'seed': seed, 'train': 50000,
          'validation': 10000, 'test': 10000, 'test_used_for_training': False,
          'files_bytes': files, 'total_bytes': sum(files.values())}
out = ROOT / 'results/setup'; out.mkdir(parents=True, exist_ok=True)
(out / 'data_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
