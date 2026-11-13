"""Build a local review ZIP from the public source allowlist; never publish."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from public_source_scope import ROOT, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix != ".zip":
        raise ValueError("--output must end in .zip")
    files = source_files()
    if not files:
        raise ValueError("Public file selection is empty")
    for name in files:
        if (ROOT / name).stat().st_size > 5 * 1024 * 1024:
            raise ValueError(f"Unexpected file over 5 MiB in source selection: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(ROOT / name, f"Tactohand/{name}")
    print(f"Source review ZIP: {output} ({len(files)} files)")


if __name__ == "__main__":
    main()
