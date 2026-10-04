#!/usr/bin/env python
"""
Split / reassemble a large file for transfer through the Colab Contents API.

`colab upload` goes through Jupyter's /api/contents, which rejects large files
(a few hundred MB fails with HTTP 400). This splits a file into parts small
enough to pass, and rebuilds it on the far side, verifying the SHA-256 so a
truncated part can never masquerade as a complete file.

    python scripts/remote/split_file.py split big.csv /tmp/parts --part-mb 32
    python scripts/remote/split_file.py join /tmp/parts /content/big.csv
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_PART_BYTES = 32 * 1024 * 1024
SUFFIX = ".part"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def split(path, out_dir, part_bytes=DEFAULT_PART_BYTES):
    path = Path(path)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    total = path.stat().st_size
    digest = sha256(path)

    parts = []
    with open(path, "rb") as fh:
        idx = 0
        while True:
            data = fh.read(part_bytes)
            if not data:
                break
            name = f"{path.stem}{SUFFIX}{idx:04d}"
            (out_dir / name).write_bytes(data)
            parts.append({"name": name, "size": len(data)})
            idx += 1

    manifest = {
        "original_name": path.name,
        "original_size": total,
        "original_sha256": digest,
        "part_bytes": part_bytes,
        "parts": parts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\n{len(parts)} parts, {total / 1e6:.1f} MB total, "
          f"largest part {max(p['size'] for p in parts) / 1e6:.1f} MB",
          file=sys.stderr)
    return manifest


def join(parts_dir, out_path, verify=True):
    parts_dir = Path(parts_dir)
    manifest_path = parts_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing - upload it alongside the parts")
    manifest = json.loads(manifest_path.read_text())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(out_path, "wb") as out:
        for part in manifest["parts"]:
            p = parts_dir / part["name"]
            if not p.exists():
                raise SystemExit(f"missing part: {p}")
            with open(p, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
            total += p.stat().st_size

    print(f"reassembled {out_path} ({total / 1e6:.1f} MB, expected "
          f"{manifest['original_size'] / 1e6:.1f} MB)")
    if total != manifest["original_size"]:
        raise SystemExit("SIZE MISMATCH - transfer was incomplete")

    if verify:
        digest = sha256(out_path)
        ok = digest == manifest["original_sha256"]
        print(f"sha256 {digest}\n       {manifest['original_sha256']}\n"
              f"       {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("CHECKSUM MISMATCH - re-upload the parts")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("path")
    s.add_argument("out_dir")
    s.add_argument("--part-mb", type=float, default=32)
    j = sub.add_parser("join")
    j.add_argument("parts_dir")
    j.add_argument("out_path")
    j.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    if a.cmd == "split":
        split(a.path, a.out_dir, int(a.part_mb * 1024 * 1024))
    else:
        join(a.parts_dir, a.out_path, verify=not a.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
