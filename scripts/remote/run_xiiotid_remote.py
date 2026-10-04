#!/usr/bin/env python
"""
Entry point executed ON a remote Colab VM (T4/L4/A100) via `colab exec -f`.

Responsibilities, in order:
  1. unpack the repository tarball that was uploaded alongside this script
  2. locate the X-IIoTID CSV (already uploaded, or fetch it from Kaggle)
  3. run the true open-set ProtoIDS experiment
  4. print a compact metrics summary and leave the artifacts in a known folder

Everything is parameterised by environment variables so the same script works
when driven by scripts/colab_t4_run.sh or run by hand in a Colab cell.

  PROTOIDS_TARBALL   path to the repo tarball        (default /content/protoids_repo.tar.gz)
  PROTOIDS_ROOT      unpack destination               (default /content/ProtoIDS)
  XIIOTID_CSV        explicit path to the CSV        (default: search PROTOIDS_ROOT/datasets/xiiotid)
  XIIOTID_WITHHELD   space-separated withheld classes (default: let the loader suggest)
  XIIOTID_EPOCHS     epochs                          (default 10)
  XIIOTID_BATCH      batch size                      (default 256)
  XIIOTID_K          prototypes per class             (default 3)
  XIIOTID_LABEL_COL  force a label column             (default: auto)
  XIIOTID_MAX_ROWS   cap CSV rows for a smoke run    (default: no cap)
  PROTOIDS_OUT       where results are written        (default PROTOIDS_ROOT/artifacts)
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/content")
TARBALL = Path(os.environ.get("PROTOIDS_TARBALL", WORK / "protoids_repo.tar.gz"))
REPO = Path(os.environ.get("PROTOIDS_ROOT", WORK / "ProtoIDS"))
OUT = Path(os.environ.get("PROTOIDS_OUT", REPO / "artifacts"))


def hr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def unpack():
    if REPO.exists() and (REPO / "src" / "protoids" / "dataset.py").exists():
        print(f"repo already present at {REPO}")
        return
    if not TARBALL.exists():
        raise SystemExit(
            f"Repo tarball not found at {TARBALL}.\n"
            f"Build it locally with:\n"
            f"  git -C <repo> archive --format=tar.gz -o {TARBALL.name} HEAD\n"
            f"and upload it with `colab upload {TARBALL.name} {TARBALL}`.")
    REPO.parent.mkdir(parents=True, exist_ok=True)
    print(f"unpacking {TARBALL} -> {REPO}")
    with tarfile.open(TARBALL) as tf:
        tf.extractall(REPO, filter="data")
    inner = REPO / "ProtoIDS" / "src" / "protoids" / "dataset.py"
    if inner.exists():
        print("archive had a ProtoIDS/ prefix; flattening")
        for child in list(REPO.iterdir()):
            if child.name == "ProtoIDS":
                continue
            shutil_move(child, REPO / "ProtoIDS")
        REPO = REPO / "ProtoIDS"
    if not (REPO / "src" / "protoids" / "dataset.py").exists():
        raise SystemExit(f"unpacked but {REPO}/src/protoids/dataset.py is missing")


def shutil_move(src, dst):
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def ensure_dependencies():
    import importlib.util
    pip = {"sklearn": "scikit-learn"}
    needed = ["torch", "numpy", "pandas", "sklearn", "joblib", "pyarrow", "matplotlib"]
    missing = [pip.get(p, p) for p in needed
               if importlib.util.find_spec(p) is None]
    if missing:
        print(f"installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def ensure_data():
    """Return a directory containing the X-IIoTID CSV."""
    explicit = os.environ.get("XIIOTID_CSV")
    if explicit and Path(explicit).is_file():
        print(f"using CSV given by XIIOTID_CSV: {explicit}")
        return Path(explicit).parent

    sys.path.insert(0, str(REPO / "src"))
    from data.download_xiiotid import ensure_xiiotid
    dest = REPO / "datasets" / "xiiotid"
    try:
        csv = ensure_xiiotid(dest_dir=str(dest), source="auto")
        return Path(csv).parent
    except Exception as exc:  # noqa: BLE001
        print(f"\ncould not obtain the dataset automatically:\n{exc}\n")
        print("Options:")
        print("  a) upload it from your machine and re-run this script:")
        print("       colab upload 'X-IIoTID dataset.csv' /content/datasets/xiiotid/")
        print("  b) give the VM Kaggle credentials, then re-run:")
        print("       colab exec   (writes ~/.kaggle/kaggle.json on the VM)")
        raise SystemExit(1)


def main():
    hr("ProtoIDS / X-IIoTID on a remote Colab VM")
    print(f"python  : {sys.version.split()[0]}")
    try:
        import torch
        print(f"torch   : {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu     : {torch.cuda.get_device_name(0)}  "
                  f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        torch = None

    unpack()
    ensure_dependencies()
    data_dir = ensure_data()

    sys.path.insert(0, str(REPO / "src"))
    from protoids.dataset import (suggest_xiiotid_withheld, _locate_xiiotid_csv,
                                  rank_xiiotid_label_columns)
    import pandas as pd

    hr("Dataset audit")
    csv = _locate_xiiotid_csv(str(data_dir))
    print(f"csv        : {csv}  ({os.path.getsize(csv) / 1e6:.1f} MB)")
    header = pd.read_csv(csv, nrows=0)
    print(f"columns    : {len(header.columns)}")
    for col, nun in rank_xiiotid_label_columns(csv, header.columns.tolist()):
        print(f"  label level {col:<8} {nun:>3} distinct")

    withheld_env = os.environ.get("XIIOTID_WITHHELD", "").strip()
    if withheld_env:
        withheld = withheld_env.split()
        print(f"\nwithheld (explicit): {withheld}")
    else:
        _, withheld, counts = suggest_xiiotid_withheld(str(data_dir),
                                                       label_col=os.environ.get("XIIOTID_LABEL_COL") or None)
        print(f"\nwithheld (suggested): {withheld}")
        print("class counts:")
        print(counts.to_frame("rows").assign(
            share=lambda d: (d.rows / d.rows.sum()).round(4)).to_string())

    hr("Training (true open-set: withhold -> fit scaler on known-train only)")
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    exp = f"protoids_xiiotid_open_k{os.environ.get('XIIOTID_K', '3')}"
    cmd = [sys.executable, str(REPO / "src" / "protoids" / "train_protoids.py"),
           "--dataset", "xiiotid",
           "--experiment_name", exp,
           "--withhold_open_set", "--withheld_classes", *withheld,
           "--num_prototypes", os.environ.get("XIIOTID_K", "3"),
           "--lambda_compact", os.environ.get("XIIOTID_LAMBDA", "0.1"),
           "--embedding_dim", os.environ.get("XIIOTID_EMBED", "32"),
           "--batch_size", os.environ.get("XIIOTID_BATCH", "256"),
           "--num_epochs", os.environ.get("XIIOTID_EPOCHS", "10"),
           "--device", device, "--seed", "42"]
    if os.environ.get("XIIOTID_LABEL_COL"):
        cmd += ["--xiiotid_label_col", os.environ["XIIOTID_LABEL_COL"]]
    if os.environ.get("XIIOTID_MAX_ROWS"):
        cmd += ["--max_rows", os.environ["XIIOTID_MAX_ROWS"]]

    env = dict(os.environ, XIIOTID_DATA_DIR=str(data_dir))
    print(" ".join(cmd), "\n", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
    print(f"\ntraining exit={rc}  elapsed={time.time() - t0:.0f}s")
    if rc != 0:
        raise SystemExit(rc)

    hr("Results")
    res = REPO / "experiments" / "results" / exp
    summary = {}
    for name in ("closed_set_results.json", "open_set_results.json",
                 "open_set_test_results.json", "xiiotid_feature_manifest.json",
                 "xiiotid_label_mapping.json"):
        p = res / name
        if not p.exists():
            continue
        data = json.load(open(p))
        summary[name] = data
        if name.endswith("_results.json"):
            flat = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            print(f"\n{name}")
            for k, v in flat.items():
                print(f"  {k:<36} {v:.4f}")

    if "open_set_test_results.json" in summary:
        t = summary["open_set_test_results.json"]
        nu_ok = t.get("unknown_samples_correctly_rejected", 0)
        nu_bad = t.get("unknown_samples_incorrectly_accepted", 0)
        k_ok = t.get("known_samples_correctly_accepted", 0)
        k_bad = t.get("known_samples_rejected_as_unknown", 0)
        print("\nconfusion of the rejection decision (test set):")
        print(f"  unknown rejected  : {nu_ok:.0f} / {nu_ok + nu_bad:.0f}"
              f"   ({nu_ok / max(nu_ok + nu_bad, 1):.4f})")
        print(f"  unknown accepted  : {nu_bad:.0f} / {nu_ok + nu_bad:.0f}"
              f"   ({nu_bad / max(nu_ok + nu_bad, 1):.4f}  <- project 'True FAR')")
        print(f"  known  rejected   : {k_bad:.0f} / {k_ok + k_bad:.0f}"
              f"   ({k_bad / max(k_ok + k_bad, 1):.4f}  <- standard open-set FAR)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts: {res}\nsummary  : {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
