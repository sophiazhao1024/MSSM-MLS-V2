"""Verify shipped files against RELEASE_MANIFEST.json (standard library only).

Run from any directory: python tools/verify_release.py
This checks integrity, not numerical accuracy or scientific reproducibility.
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file():
            failures.append(f"Missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Changed: {name}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(manifest['files'])} shipped files. No physics calculation performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
