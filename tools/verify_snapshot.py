#!/usr/bin/env python3
"""Verify publication source and optional offline wheel checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def unlisted_files(root: Path, manifest_path: Path, manifest: dict) -> list[str]:
    """Identify unexpected source files without scanning local environments."""
    unexpected = []
    for directory, folders, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in tuple(folders):
            path = parent / name
            if (parent == root and name in {".git", ".venv", "offline"}) or name == "__pycache__":
                folders.remove(name)
            elif path.is_symlink():
                unexpected.append(path.relative_to(root).as_posix())
                folders.remove(name)
        for name in files:
            path = parent / name
            if path != manifest_path and path.relative_to(root).as_posix() not in manifest:
                unexpected.append(path.relative_to(root).as_posix())
    return sorted(unexpected)


def verify(root: Path, manifest_path: Path, *, strict: bool = False) -> int:
    """Check the listed regular files against their SHA-256 digests."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {manifest_path}")
    for name, expected in manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(p in {".", ".."} for p in relative.parts):
            raise ValueError(f"Invalid manifest path: {name}")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe file: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {path}")
    if strict:
        unexpected = unlisted_files(root, manifest_path, manifest)
        if unexpected:
            raise ValueError(f"Unlisted files: {', '.join(unexpected)}")
    return len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", action="store_true", help="also verify the offline wheel collection")
    parser.add_argument("--strict", action="store_true", help="reject unlisted source files")
    args = parser.parse_args()
    count = verify(ROOT, ROOT / "SNAPSHOT-SHA256.json", strict=args.strict)
    print(f"Verified {count} source snapshot files")
    if args.wheels:
        wheel_root = ROOT / "offline/wheels"
        count = verify(wheel_root, wheel_root / "SHA256.json", strict=True)
        print(f"Verified {count} offline wheels")


if __name__ == "__main__":
    main()
