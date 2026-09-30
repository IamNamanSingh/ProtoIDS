"""Cheap smoke test: does a remote T4 actually appear, with torch working?

    colab run --gpu T4 scripts/remote/smoke_gpu.py
"""
import sys

print("python:", sys.version.split()[0], flush=True)

try:
    import torch
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch

print("torch :", torch.__version__, flush=True)
print("cuda  :", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu   :", torch.cuda.get_device_name(0), flush=True)
    print("vram  : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9),
          flush=True)
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul:", float((x @ x).mean()), flush=True)

import os
print("cwd   :", os.getcwd(), flush=True)
print("disk  :", flush=True)
os.system("df -h /content | tail -1")
print("SMOKE OK", flush=True)
