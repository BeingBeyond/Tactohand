from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def write_extra_parquet(
    output_root: Path,
    name: str,
    rows: List[dict],
    target_bytes: int,
    overwrite: bool = False,
) -> List[dict]:
    if not rows:
        return []

    out_dir = output_root / "extra" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    chunk_idx = 0
    file_idx = 0
    current_size = 0
    avg_row_size = 2000
    buffer: List[dict] = []

    pbar = tqdm(total=len(rows), desc=f"Writing extra {name}", unit="row")

    for row in rows:
        buffer.append(row)
        current_size += avg_row_size
        pbar.update(1)

        if current_size >= target_bytes and buffer:
            out_path = out_dir / f"extra-{file_idx:03d}.parquet"
            if out_path.exists() and not overwrite:
                pass
            else:
                table = pa.Table.from_pylist(buffer)
                # Optimized PyArrow write parameters
                pq.write_table(
                    table,
                    out_path,
                    compression='zstd',
                    compression_level=3,
                    use_dictionary=True,
                    write_statistics=False,
                    row_group_size=100000,
                )
                actual_size = out_path.stat().st_size
                avg_row_size = actual_size / max(len(buffer), 1)

            files.append({
                "name": name,
                "chunk": chunk_idx,
                "file": file_idx,
                "rows": len(buffer),
            })

            buffer = []
            current_size = 0
            file_idx += 1
            if file_idx >= 1000:
                chunk_idx += 1
                file_idx = 0

    if buffer:
        out_path = out_dir / f"extra-{file_idx:03d}.parquet"
        if out_path.exists() and not overwrite:
            pass
        else:
            table = pa.Table.from_pylist(buffer)
            # Optimized PyArrow write parameters
            pq.write_table(
                table,
                out_path,
                compression='zstd',
                compression_level=3,
                use_dictionary=True,
                write_statistics=False,
                row_group_size=100000,
            )

        files.append({
            "name": name,
            "chunk": chunk_idx,
            "file": file_idx,
            "rows": len(buffer),
        })

    pbar.close()
    return files
