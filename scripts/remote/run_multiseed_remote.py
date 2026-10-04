#!/usr/bin/env python
"""Run the multi-seed measurement on a remote Colab VM."""
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content")
REPO = Path(os.environ.get("PROTOIDS_ROOT", WORK / "ProtoIDS"))
TAG = os.environ.get("SESSION_TAG", "protoids-xiiotid")
PROTOCOLS = os.environ.get("XIIOTID_PROTOCOLS", "s1,s2,fam").split(",")
SEEDS = os.environ.get("XIIOTID_SEEDS", "42 7 1234 5 99 2024").split()
KS = os.environ.get("XIIOTID_KS", "1 3").split()
EPOCHS = os.environ.get("XIIOTID_EPOCHS", "10")
BATCH = os.environ.get("XIIOTID_BATCH", "256")
OUT = Path(os.environ.get("PROTOIDS_OUT", REPO / "experiments" / "results" / f"multiseed_{TAG}"))

WITHHELD = {
    "s1": ["Scanning_vulnerability", "Dictionary", "Reverse_shell"],
    "s2": ["Exfiltration", "BruteForce", "Modbus_register_reading"],
    "fam": ["C&C", "Exfiltration", "MQTT_cloud_broker_subscription",
            "Modbus_register_reading", "TCP Relay"],
}


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def run(args, label):
    print(f"\n$ {' '.join(str(a) for a in args)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(a) for a in args], cwd=str(REPO)).returncode
    print(f"  -> {label} exit={rc} in {time.time() - t0:.0f}s", flush=True)
    return rc


def main():
    hr("ProtoIDS multi-seed on Colab")
    print(f"repo      : {REPO}")
    print(f"protocols : {PROTOCOLS}")
    print(f"seeds     : {SEEDS}")
    print(f"K values  : {KS}")
    try:
        import torch
        print(f"gpu       : {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        pass

    data = WORK / "datasets" / "xiiotid"
    print(f"data      : {data}")

    hr("stage 1/2: build cached protocol splits")
    for tag in PROTOCOLS:
        proc = REPO / "experiments" / "processed" / f"xiiotid_{tag}"
        if (proc / "split_meta.json").exists():
            print(f"  {tag}: cached")
            continue
        print(f"  {tag}: building (withheld={WITHHELD[tag]})")
        rc = run([sys.executable, REPO / "src/protoids/train_protoids.py",
                  "--dataset", "xiiotid", "--withhold_open_set",
                  "--withheld_classes", *WITHHELD[tag],
                  "--experiment_name", f"prep_{tag}",
                  "--num_prototypes", "1", "--embedding_dim", "64",
                  "--num_epochs", "1", "--batch_size", BATCH, "--device", "cuda", "--seed", "42"],
                 f"prep_{tag}")
        if rc:
            print(f"  {tag}: prep FAILED")
            continue
        src = REPO / "experiments" / "processed" / f"prep_{tag}"
        if src.exists():
            src.rename(proc)

    import json
    for tag in PROTOCOLS:
        p = REPO / "experiments" / "processed" / f"xiiotid_{tag}" / "split_meta.json"
        if p.exists():
            m = json.load(open(p))
            print(f"  {tag}: dim={m['input_dim']} known={len(m['known_classes'])} "
                  f"rows={m['counts']}")
        else:
            print(f"  {tag}: MISSING")

    hr("stage 2/2: multi-seed measurement")
    cmd = [sys.executable, "-u", "-m", "protoids.multiseed"]
    for t in PROTOCOLS:
        cmd += ["--processed", f"{t}={REPO / 'experiments' / 'processed' / f'xiiotid_{t}'}"]
    cmd += ["--seeds", *[str(s) for s in SEEDS],
            "--configs", *[str(k) for k in KS],
            "--epochs", EPOCHS, "--batch-size", BATCH,
            "--device", "cuda", "--out", OUT, "--resume"]
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    print(f"\nmultiseed exit={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)

    hr("summary")
    s = OUT / "summary.json"
    if s.exists():
        print(open(s).read()[:4000])
    else:
        print("no summary.json produced", flush=True)
    print(f"\nartifacts: {OUT}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
