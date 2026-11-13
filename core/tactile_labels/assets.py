"""Object asset name tables: join episode object-instance ids to names.

``label_index.object_ids`` entries are per-episode instance ids whose
format varies per dataset family (gigahands
``ep:12:<cat>:<ann>:17_instruments/flute_scan/flute.obj:0``, dexycb
``dexycb:1|<session>:ycb:002_master_chef_can``), so they cannot group
one physical object across episodes.  ``objects/assets.parquet`` in the
processed root is the asset-level table; the join is exact on
``object_asset_id`` first (some datasets embed it verbatim in the oid),
then on the longest ``source_id`` contained in the oid.

Shared by the training reader's object allowlist and the review
visualizers so a picked display name means the same thing in both.
"""
from __future__ import annotations
import core.data.processed_compat as processed_contract
from core.geometry.object_assets import read_asset_table

from pathlib import Path

import pyarrow.parquet as pq


def load_asset_names(assets_parquet: Path,
                     ) -> tuple[dict[str, str], dict[str, str]]:
    """``(object_asset_id -> name, source_id -> name)`` tables."""
    if not assets_parquet.exists() and processed_contract.is_processed(assets_parquet.parent.parent):
        assets_parquet = processed_contract.object_path(assets_parquet.parent.parent,'assets_path','objects/assets.parquet')
    rows = read_asset_table(assets_parquet, 'assets').to_pylist()
    asset_names = {str(row["object_asset_id"]): str(row["name"])
                   for row in rows}
    source_names = {str(row["source_id"]): str(row["name"])
                    for row in rows}
    return asset_names, source_names


def object_display_name(oid: str, asset_names: dict[str, str],
                        source_names: dict[str, str]) -> str:
    """Map one episode object-instance id to its asset display name."""
    name = asset_names.get(oid)
    if name is None:
        name = source_names.get(
            max((s for s in source_names if s in oid), key=len,
                default=""), oid)
    return name
