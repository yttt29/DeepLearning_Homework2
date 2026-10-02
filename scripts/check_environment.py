"""Verify dependencies, plotting, CPU/MPS training operations and a Jupyter kernel."""
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
import sklearn
import tqdm
import torchvision
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jupyter_client import KernelManager

OUT = Path(__file__).resolve().parents[1] / 'results/setup'
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(4)
base = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
x = torch.randn(256, 784)
y = torch.randint(10, (256,))
reference = base(x).detach()
results = {}
for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
    import copy
    model = copy.deepcopy(base).to(device)
    xx, yy = x.to(device), y.to(device)
    torch.testing.assert_close(model(xx).detach().cpu(), reference, atol=2e-5, rtol=2e-4)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    def step():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xx), yy)
        loss.backward()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters())
        opt.step()
        return loss
    initial = float(torch.nn.functional.cross_entropy(model(xx), yy).item())
    for _ in range(5): step()
    if device == 'mps': torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(30): loss = step()
    if device == 'mps': torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    final = float(loss.item())
    assert final < initial
    results[device] = {'initial_loss': initial, 'final_loss': final, 'seconds_30_steps': elapsed}
plt.plot([0, 1, 2], [0, 1, 4]); plt.title('Environment plotting check')
plt.savefig(OUT / 'plot_check.png'); plt.close()
km = KernelManager(kernel_name='dlcv-lab1')
km.start_kernel()
client = km.client(); client.start_channels()
try:
    client.wait_for_ready(timeout=30)
    reply = client.execute_interactive("import sys, torch; assert sys.prefix.endswith('/envs/dlcv-lab1'); print(sys.executable)", timeout=30)
    assert reply['content']['status'] == 'ok', reply
finally:
    client.stop_channels(); km.shutdown_kernel(now=True)
report = {'python': sys.version, 'executable': sys.executable, 'architecture': platform.machine(),
          'versions': {p: metadata.version(p) for p in ['torch', 'torchvision', 'numpy', 'matplotlib', 'pandas', 'scikit-learn', 'tqdm', 'ipykernel']},
          'mps_available': torch.backends.mps.is_available(), 'devices': results, 'kernel_passed': True,
          'timing_note': 'Synthetic smoke test includes gradient checks and synchronization overhead; not a training benchmark.'}
(OUT / 'environment_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
