"""
X-IIoTID dataset acquisition helper (works locally AND on Google Colab).

The X-IIoTID dataset (X-IIoTID: A Connectivity- and Device-agnostic Intrusion
Dataset for Industrial Internet of Things) is published by the authors on Kaggle:

    ref:  munaalhawawreh/xiiotid-iiot-intrusion-dataset
    file: X-IIoTID Dataset.csv   (~355 MB, 820,834 rows, 42 network features,
                                 3 label levels)

The same paper/version is mirrored on IEEE DataPort, but IEEE DataPort requires an
institutional login, so Kaggle is the practical automated source. Kaggle accounts
are free; you only need an API token.

Supported sources (auto-detected in this order when ``--source auto``):

    existing  a CSV is already present in the destination directory
    kaggle    Kaggle API, via ``kagglehub`` if installed, else the ``kaggle`` CLI
    url       direct HTTP(S) download from ``--url``
    gdrive    copy from a mounted Google Drive path (Colab)

Authentication on Colab uses Colab Secrets (KAGGLE_USERNAME / KAGGLE_KEY) and falls
back to plain environment variables or ``~/.kaggle/kaggle.json`` anywhere else.

Usage
-----
    python -m data.download_xiiotid --dest datasets/xiiotid

    python -m data.download_xiiotid --source kaggle --dest datasets/xiiotid
    python -m data.download_xiiotid --source url --url https://.../X-IIoTID.csv
    python -m data.download_xiiotid --source gdrive --gdrive-path /content/drive/MyDrive/xiiotid

From a notebook::

    import sys; sys.path.insert(0, "src")
    from data.download_xiiotid import ensure_xiiotid
    csv_path = ensure_xiiotid(dest_dir="datasets/xiiotid")
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from typing import Optional, List


KAGGLE_REF = "munaalhawawreh/xiiotid-iiot-intrusion-dataset"
KAGGLE_FILE = "X-IIoTID Dataset.csv"
EXPECTED_ARCHIVE_BYTES = 355_308_902
EXPECTED_ROWS = 820_834
EXPECTED_COLUMNS = 49  # 42 features + 3 label levels + id/timestamp/ip/port/...

DEFAULT_DEST = os.path.join("datasets", "xiiotid")

_CSV_HINTS = ("x-iiotid", "xiiotid", "x-iiot")


def _log(msg: str) -> None:
    print(f"[xiiotid] {msg}", flush=True)


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _secret(name: str) -> Optional[str]:
    """Read a value from Colab Secrets, else from the environment."""
    if _is_colab():
        try:
            from google.colab import userdata
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass  # secret not set, or not authorised yet
    return os.environ.get(name) or None


def _write_kaggle_credentials(username: Optional[str] = None,
                              key: Optional[str] = None) -> bool:
    """
    Materialise ``~/.kaggle/kaggle.json`` so the ``kaggle`` CLI and
    ``kagglehub`` both work. Returns True if credentials are available.
    """
    username = username or _secret("KAGGLE_USERNAME")
    key = key or _secret("KAGGLE_KEY")

    if not (username and key):
        if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            _log("Using existing ~/.kaggle/kaggle.json")
            return True
        _log("No Kaggle credentials found.")
        return False

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as fh:
        json.dump({"username": username, "key": key}, fh)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    _log(f"Wrote Kaggle credentials for user '{username}'")
    return True


def _download_from_kaggle(dest_dir: str, force: bool = False) -> Optional[str]:
    """Download via kagglehub, falling back to the kaggle CLI. Returns the CSV path."""
    if not _write_kaggle_credentials():
        return None

    if not force and _find_csv(dest_dir) is not None:
        return _find_csv(dest_dir)

    os.makedirs(dest_dir, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        kagglehub = None

    if kagglehub is not None:
        _log("Downloading via kagglehub (this pulls ~355 MB)...")
        try:
            cached = kagglehub.dataset_download(KAGGLE_REF)
            found = _find_csv(cached) or _find_csv(dest_dir)
            if found is None:
                candidate = _find_csv(cached, require_hint=False)
                if candidate is None:
                    for name in os.listdir(cached):
                        if name.lower().endswith(".csv"):
                            candidate = os.path.join(cached, name)
                            break
                if candidate is None:
                    raise FileNotFoundError(f"No CSV inside {cached}")
                target = os.path.join(dest_dir, KAGGLE_FILE)
                shutil.copy2(candidate, target)
                found = target
            if os.path.dirname(os.path.abspath(found)) != os.path.abspath(dest_dir):
                target = os.path.join(dest_dir, os.path.basename(found))
                if not os.path.exists(target):
                    shutil.copy2(found, target)
                found = target
            _log(f"kagglehub OK -> {found}")
            return found
        except Exception as exc:  # noqa: BLE001 - fall through to the CLI
            _log(f"kagglehub failed ({exc}); trying the kaggle CLI...")

    try:
        import subprocess
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
               "-d", KAGGLE_REF, "-p", dest_dir, "--unzip", "--force"]
        _log("Running: " + " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            _log(f"kaggle CLI exited with {proc.returncode}")
            return None
    except ImportError:
        _log("Neither kagglehub nor the kaggle CLI is installed. "
             "Run: pip install kagglehub kaggle")
        return None

    found = _find_csv(dest_dir)
    if found:
        _log(f"kaggle CLI OK -> {found}")
    return found


def _download_from_url(url: str, dest_dir: str, force: bool = False) -> Optional[str]:
    """Stream a direct HTTP(S) URL to <dest_dir>/X-IIoTID Dataset.csv."""
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, KAGGLE_FILE)
    if os.path.exists(target) and not force:
        _log(f"Already present: {target}")
        return target

    _log(f"Downloading {url}")
    tmp = target + ".part"
    start = time.time()
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_report = 64 * 1024 * 1024
        while True:
            block = resp.read(1024 * 1024)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if done >= next_report:
                pct = f"{100.0 * done / total:.0f}%" if total else "?"
                _log(f"  {done / 1e6:.0f} MB ({pct})")
                next_report += 64 * 1024 * 1024
    os.replace(tmp, target)
    _log(f"Saved {target} ({os.path.getsize(target) / 1e6:.1f} MB "
         f"in {time.time() - start:.0f}s)")
    return target


def _copy_from_gdrive(gdrive_path: str, dest_dir: str, force: bool = False) -> Optional[str]:
    """Copy an already-downloaded CSV out of a mounted Google Drive folder."""
    src = gdrive_path
    if os.path.isdir(src):
        found = _find_csv(src)
        if found is None:
            raise FileNotFoundError(f"No X-IIoTID CSV found under {src}")
        src = found
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{src} does not exist (is Drive mounted?)")

    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, KAGGLE_FILE)
    if os.path.exists(target) and not force:
        _log(f"Already present: {target}")
        return target
    _log(f"Copying {src} -> {target} (Drive is slow, be patient)")
    shutil.copy2(src, target)
    return target


def _find_csv(root: str, require_hint: bool = True) -> Optional[str]:
    """Recursively look for the X-IIoTID CSV under ``root``."""
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            low = name.lower()
            if not low.endswith(".csv"):
                continue
            if not require_hint or any(h in low for h in _CSV_HINTS):
                return os.path.join(dirpath, name)
    return None


def verify_csv(csv_path: str, read_rows: bool = True) -> dict:
    """
    Cheap sanity check of the downloaded file. Never raises; returns a report.
    """
    report = {
        "path": csv_path,
        "exists": os.path.isfile(csv_path),
        "size_bytes": os.path.getsize(csv_path) if os.path.isfile(csv_path) else 0,
    }
    if not report["exists"]:
        return report

    try:
        import pandas as pd
        header = pd.read_csv(csv_path, nrows=0)
        report["n_columns"] = len(header.columns)
        report["columns"] = header.columns.tolist()
        if read_rows:
            rows = 0
            with open(csv_path, "rb") as fh:
                for _ in fh:
                    rows += 1
            report["n_rows"] = max(rows - 1, 0)  # minus header
            report["rows_match_expected"] = abs(report["n_rows"] - EXPECTED_ROWS) <= 1
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
    return report


def _write_metadata(dest_dir: str, csv_path: str, report: dict, source: str) -> str:
    """Record provenance so results can cite exactly what was used."""
    meta = {
        "dataset": "X-IIoTID",
        "paper": ("X-IIoTID: A Connectivity- and Device-agnostic Intrusion Dataset "
                  "for Industrial Internet of Things"),
        "kaggle_ref": KAGGLE_REF,
        "kaggle_file": KAGGLE_FILE,
        "expected_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "expected_rows": EXPECTED_ROWS,
        "expected_columns": EXPECTED_COLUMNS,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acquisition_source": source,
        "csv_path": os.path.relpath(csv_path),
        "size_bytes": report.get("size_bytes"),
        "n_rows": report.get("n_rows"),
        "n_columns": report.get("n_columns"),
        "verification_error": report.get("error"),
    }
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "download_metadata.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


def ensure_xiiotid(dest_dir: str = DEFAULT_DEST,
                   source: str = "auto",
                   url: Optional[str] = None,
                   gdrive_path: Optional[str] = None,
                   force: bool = False,
                   write_metadata: bool = True) -> str:
    """
    Make sure the X-IIoTID CSV is on disk and return its path.

    Raises RuntimeError with actionable guidance if no source succeeds.
    """
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    csv_path: Optional[str] = None
    used_source = source

    if source == "existing":
        csv_path = _find_csv(dest_dir)
        if csv_path is None:
            raise RuntimeError(f"No X-IIoTID CSV found under {dest_dir}")
    elif source == "url":
        if not url:
            raise ValueError("--source url requires --url")
        csv_path = _download_from_url(url, dest_dir, force=force)
    elif source == "gdrive":
        if not gdrive_path:
            raise ValueError("--source gdrive requires --gdrive-path")
        csv_path = _copy_from_gdrive(gdrive_path, dest_dir, force=force)
    elif source == "kaggle":
        csv_path = _download_from_kaggle(dest_dir, force=force)
    elif source == "auto":
        csv_path = _find_csv(dest_dir)
        if csv_path is not None:
            used_source = "existing"
            _log(f"Found existing CSV: {csv_path}")
        else:
            _log("No local CSV found, trying Kaggle...")
            csv_path = _download_from_kaggle(dest_dir, force=force)
            used_source = "kaggle"
    else:
        raise ValueError(f"Unknown source: {source}")

    if csv_path is None:
        raise RuntimeError(
            "Could not obtain X-IIoTID. Do one of:\n"
            "  1. Free Kaggle account -> Settings -> Create API Token, then on Colab\n"
            "     add secrets KAGGLE_USERNAME and KAGGLE_KEY, or locally write\n"
            "     ~/.kaggle/kaggle.json and run:\n"
            "       pip install kagglehub kaggle\n"
            "       python -m data.download_xiiotid --source kaggle\n"
            "  2. Download 'X-IIoTID Dataset.csv' manually from\n"
            f"     https://www.kaggle.com/datasets/{KAGGLE_REF}\n"
            f"     and drop it into {dest_dir}\n"
            "  3. IEEE DataPort mirror (institutional login):\n"
            "     https://ieee-dataport.org/open-access/x-iiotid\n"
            "     then: python -m data.download_xiiotid --source gdrive \\\n"
            "            --gdrive-path /path/to/folder"
        )

    report = verify_csv(csv_path)
    if write_metadata:
        meta_path = _write_metadata(dest_dir, csv_path, report, used_source)
        _log(f"Wrote provenance: {meta_path}")

    _log(f"CSV: {csv_path}")
    _log(f"  size   : {report.get('size_bytes', 0) / 1e6:.1f} MB")
    if report.get("n_rows"):
        _log(f"  rows   : {report['n_rows']} (expected {EXPECTED_ROWS})")
    if report.get("n_columns"):
        _log(f"  columns: {report['n_columns']} (expected {EXPECTED_COLUMNS})")
    if report.get("error"):
        _log(f"  WARNING: header/row check failed: {report['error']}")

    return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the X-IIoTID dataset (Kaggle / URL / Drive).")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "existing", "kaggle", "url", "gdrive"],
                        help="where to get the data (default: auto)")
    parser.add_argument("--url", default=None, help="direct URL for --source url")
    parser.add_argument("--gdrive-path", default=None,
                        help="file or folder in mounted Drive for --source gdrive")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the row/column sanity check")
    args = parser.parse_args(argv)

    try:
        ensure_xiiotid(dest_dir=args.dest, source=args.source, url=args.url,
                       gdrive_path=args.gdrive_path, force=args.force)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        _log(f"FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    raise SystemExit(main())
