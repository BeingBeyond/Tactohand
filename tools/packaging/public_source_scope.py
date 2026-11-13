"""Shared, bounded file selection for wheels and source distributions."""
from __future__ import annotations

from pathlib import Path
from fnmatch import fnmatchcase
import tomllib


ROOT = Path(__file__).resolve().parents[2]
POLICY = tomllib.loads((ROOT / "packaging/public_files.toml").read_text())


def _matches(parts: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return _matches(parts, pattern[1:]) or bool(parts) and _matches(parts[1:], pattern)
    return (bool(parts) and fnmatchcase(parts[0], pattern[0])
            and _matches(parts[1:], pattern[1:]))


def allows_wheel(path: str) -> bool:
    # Match from the archive root; '*' never swallows a directory separator.
    parts = tuple(path.split("/"))
    return (any(_matches(parts, tuple(p.split("/"))) for p in POLICY["wheel"]["include"])
            and not any(_matches(parts, tuple(p.split("/"))) for p in POLICY["wheel"]["exclude"]))


def source_files() -> list[str]:
    selected: set[str] = set()
    for pattern in POLICY["wheel"]["include"]:
        for path in ROOT.glob(pattern):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(ROOT).as_posix()
                if allows_wheel(relative):
                    selected.add(relative)
    for pattern in POLICY["source"]["include"]:
        for path in ROOT.glob(pattern):
            if path.is_file() and not path.is_symlink():
                selected.add(path.relative_to(ROOT).as_posix())
    return sorted(selected)
