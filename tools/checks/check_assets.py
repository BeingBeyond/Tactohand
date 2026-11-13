#!/usr/bin/env python3
"""Verify local asset fingerprints and links without loading pickle payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_assets(root: Path, include_archive: bool = False) -> dict:
    manifest = json.loads((root / "assets/manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported asset manifest schema")
    files = dict(manifest["files"])
    if include_archive:
        files.update(manifest["archived"])
    checked, skipped, failures = [], [], []
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            if expected.get("required", include_archive):
                failures.append(f"missing required asset: {relative}")
            else:
                skipped.append(relative)
            continue
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            failures.append(f"content mismatch: {relative}")
        else:
            checked.append(relative)
    checked_links = []
    for relative, expected in manifest["links"].items():
        path = root / relative
        if not path.exists() and not path.is_symlink() and not expected["required"]:
            skipped.append(relative)
            continue
        if not path.is_symlink() or str(path.readlink()) != expected["target"] or not path.exists():
            failures.append(f"missing, changed or broken link: {relative}")
        else:
            checked_links.append(relative)
    registered = set(manifest["files"]) | set(manifest["links"])
    for path in (root / "assets").rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in registered or path.name == "README.md" or relative == "assets/manifest.json":
            continue
        failures.append(f"unregistered asset: {relative}")
    return {"passed": not failures, "checked_files": checked,
            "checked_links": checked_links, "skipped_optional": skipped,
            "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-archive", action="store_true",
                        help="also verify this checkout's relocated downloads/retired resources")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_assets(ROOT, args.include_archive)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Assets: {len(report['checked_files'])} files, "
          f"{len(report['checked_links'])} links, "
          f"{len(report['skipped_optional'])} optional absent, "
          f"{len(report['failures'])} failures")
    for message in report["failures"][:50]:
        print(message)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
