"""One MANO loading boundary; basis changes never convert stored parameters.

Canonical fitting/consumption uses corrected, full axis angle, a flat mean,
no centering, and meters (manopth alone returns millimeters). Native dataset
adapters must name their source basis explicitly. No caller should mutate
shapedirs after loading. Only fingerprinted stock/corrected assets are accepted.
"""
from __future__ import annotations
import core.data.processed_compat as processed_contract

from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STOCK_LEFT = "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30"
CORRECTED_LEFT = "70a2ffca4f09bef54a12b0a3ce078df456e2646bd283f3c55bd1bf5e2b5b5922"
RIGHT = "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"
CONFIRMED = {"user_confirmed", "human_reviewed",
             "code_confirmed_with_installed_lineage",
             "code_confirmed_sampled_native_lineage"}
REGISTRY = REPO / "artifacts/tactohand/mano_basis_migration_20260928/basis_decisions.json"


def migration_error_limit(registry: dict, dataset: str) -> float:
    """Return the audited per-dataset surface gate, defaulting to 1 mm."""
    decision = registry['datasets'][dataset]
    policy = decision.get('precision_acceptance', {})
    limit = float(policy.get('max_vertex_error_mm',
                             registry.get('strict_max_vertex_error_mm', 1.0)))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f'{dataset}: invalid migration error limit')
    return limit


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compat() -> None:
    import numpy as np
    if not hasattr(inspect, "getargspec"):
        from collections import namedtuple
        arg_spec = namedtuple("ArgSpec", "args varargs keywords defaults")
        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return arg_spec(spec.args, spec.varargs, spec.varkw, spec.defaults)
        setattr(inspect, "getargspec", getargspec)
    for name, value in {"bool": bool, "int": int, "float": float,
                        "complex": complex, "object": object,
                        "unicode": str, "str": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


@lru_cache(maxsize=32)
def _asset_basis(directory: str, left_size: int, left_mtime: int,
                 right_size: int, right_mtime: int) -> str:
    root = Path(directory)
    left_hash = file_sha256(root / "MANO_LEFT.pkl")
    if file_sha256(root / "MANO_RIGHT.pkl") != RIGHT:
        raise ValueError(f"Unrecognized MANO_RIGHT fingerprint: {root}")
    if left_hash == STOCK_LEFT:
        return "stock"
    if left_hash == CORRECTED_LEFT:
        return "corrected"
    raise ValueError(f"Unrecognized MANO_LEFT fingerprint: {root}; register evidence before use")


def asset_directory(root: Path) -> tuple[Path, str]:
    root = root.expanduser().resolve()
    directory = root / "models" if (root / "models/MANO_LEFT.pkl").is_file() else root
    left, right = directory / "MANO_LEFT.pkl", directory / "MANO_RIGHT.pkl"
    a, b = left.stat(), right.stat()
    return directory, _asset_basis(str(directory), a.st_size, a.st_mtime_ns,
                                  b.st_size, b.st_mtime_ns)


def create_mano_layer(*, backend: str = "manotorch", basis: str = "corrected",
                      assets_root: Path | str | None = None, side: str = "right",
                      **kwargs):
    """Load a fresh module, normalizing the verified PKL to the requested basis.

    Passing stock files with basis=corrected flips once; passing corrected files
    with the same request does not flip. Reverse decoding for audits is explicit.
    The native backend API/output units remain intact.
    """
    if basis not in {"stock", "corrected"}:
        raise ValueError("basis must be explicitly stock or corrected")
    roots = [kwargs.pop(key) for key in ("mano_root", "mano_assets_root", "model_path")
             if key in kwargs]
    if len(roots) + (assets_root is not None) > 1:
        raise ValueError("Specify only one MANO asset root")
    default = REPO / "assets" / "mano" / basis
    root = Path(assets_root or (roots[0] if roots else os.environ.get("MANO_ASSETS_ROOT", str(default))))
    directory, stored_basis = asset_directory(root)
    _compat()
    kwargs.setdefault("use_pca", False)
    kwargs.setdefault("flat_hand_mean", True)
    if backend == "smplx":
        from smplx import MANO
        side = "right" if kwargs.pop("is_rhand", side == "right") else "left"
        layer = MANO(model_path=str(directory), is_rhand=side == "right", **kwargs)
        shapedirs = layer.shapedirs
    else:
        kwargs.setdefault("center_idx", None)
        if backend == "manotorch":
            from manotorch.manolayer import ManoLayer
            # manotorch requires <assets>/models; manopth takes the models directory.
            if directory.name != "models":
                raise ValueError("manotorch requires an asset root with models/MANO_*.pkl")
            layer = ManoLayer(side=side, mano_assets_root=str(directory.parent), **kwargs)
        elif backend == "manopth":
            from manopth.manolayer import ManoLayer
            layer = ManoLayer(side=side, mano_root=str(directory), **kwargs)
        else:
            raise ValueError(f"Unknown MANO backend: {backend}")
        shapedirs = layer.th_shapedirs
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if side == "left" and stored_basis != basis:
        import torch
        with torch.no_grad():
            shapedirs[:, 0, :].mul_(-1)
    layer.mano_basis = basis
    layer.mano_backend = backend
    return layer


def ManoTorchLayer(**kwargs):
    return create_mano_layer(backend="manotorch", **kwargs)


def ManoPthLayer(**kwargs):
    return create_mano_layer(backend="manopth", **kwargs)


def NativeManoTorchLayer(**kwargs):
    kwargs.setdefault("basis", "stock")
    return create_mano_layer(backend="manotorch", **kwargs)


def NativeManoPthLayer(**kwargs):
    kwargs.setdefault("basis", "stock")
    return create_mano_layer(backend="manopth", **kwargs)


def NativeSMPLXMANO(**kwargs):
    kwargs.setdefault("basis", "stock")
    return create_mano_layer(backend="smplx", **kwargs)


def contract_from_info(info: dict) -> dict | None:
    contract = info.get("mano_contract")
    if contract is None:
        return None
    if (contract.get("basis") != "corrected" or contract.get("version") != 1
            or contract.get("flat_hand_mean") is not True
            or contract.get("center_idx") is not None
            or contract.get("pose") != "full_axis_angle_48"
            or contract.get("translation_unit") != "m"
            or contract.get("assets_sha256") != {"LEFT": CORRECTED_LEFT, "RIGHT": RIGHT}):
        raise ValueError("Invalid or incompatible canonical MANO contract")
    return contract


def source_basis_from_info(info: dict) -> str | None:
    """Persistent source decoding choice, distinct from a corrected migration."""
    decision = info.get("mano_source_basis")
    if decision is None:
        return None
    basis = decision.get("basis")
    expected_left = STOCK_LEFT if basis == "stock" else CORRECTED_LEFT
    if (decision.get("version") != 1 or basis not in {"stock", "corrected"}
            or decision.get("decision_status") not in CONFIRMED
            or decision.get("assets_sha256") != {"LEFT": expected_left, "RIGHT": RIGHT}):
        raise ValueError("Invalid persistent MANO source basis decision")
    return basis


def basis_for_dataset(root: Path, *, side: str = "left") -> str:
    """Use installed contract, otherwise a hash-checked audited source decision."""
    info_path = root / "meta/info.json"
    info = processed_contract.read_info(root)
    if info.get("beingh_human", {}).get("capabilities", {}).get("hmr", {}).get("mano") is False:
        raise ValueError(f"{root}: MANO is not declared; use the native geometry adapter")
    if contract_from_info(info) is not None:
        return "corrected"
    source_basis = source_basis_from_info(info)
    if source_basis is not None:
        return "corrected" if side == "right" else source_basis
    registry_path = Path(os.environ.get("MANO_BASIS_DECISIONS", str(REGISTRY))).expanduser()
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"{root}: no installed mano_contract or mano_source_basis, and "
            f"audited MANO decisions are missing at {registry_path}. "
            "Use data with an installed MANO contract or set "
            "MANO_BASIS_DECISIONS to a reviewed registry; "
            "do not infer the source basis from the dataset name.")
    registry = json.loads(registry_path.read_text())
    decision = registry["datasets"].get(root.name.removesuffix("_lerobot"), {})
    if decision.get("source_info_sha256") != (info.get('_source_info_sha256') or file_sha256(info_path)):
        raise ValueError(f"MANO decision metadata changed: {root}; re-audit or install a contract")
    if decision.get("decision_status") in CONFIRMED and decision.get("source_basis") in {"stock", "corrected"}:
        # Right assets are identical, but an unresolved native parameter chain
        # must still not be silently treated as a valid right MANO annotation.
        return "corrected" if side == "right" else decision["source_basis"]
    if decision.get("decision_status") in {"no_semantic_left_mano_full_census", "right_model_instances_no_left_basis_change"}:
        return "corrected"
    raise ValueError(f"Unresolved MANO source basis: {root}; source geometry adapter or evidence required")


@lru_cache(maxsize=256)
def _checked_file_digest(path: str, size: int, mtime_ns: int) -> str:
    return file_sha256(Path(path))


def validate_migrated_hmr_file(root: Path, relative: str) -> None:
    """Reject a later rewrite of a migrated HMR file, including legacy writers."""
    info = processed_contract.read_info(root)
    contract = contract_from_info(info)
    if contract is None or contract.get('source_basis') != 'stock':
        return
    if processed_contract.is_processed(root):
        reader = processed_contract.reader_for(root)
        layout = reader.manifest.get('parquet_layout', {})
        if (layout.get('order') == 'unchanged'
                and layout.get('offsets') == 'absolute_file_rows'
                and layout.get('run')
                and layout.get('max_row_group_rows') == 4096
                and layout.get('validation') == 'footer_only'):
            # This publication re-encoded Parquet without refreshing the
            # migration-era byte digests. They cannot certify current bytes;
            # this known publication state is documented, not a runtime error.
            return
        table = reader.table('integrity')
        rows = [r for r in table.to_pylist() if r['path'] == relative]
        if len(rows) != 1 or rows[0]['geometry_revision'] != contract['geometry_revision']:
            raise ValueError('Migrated MANO integrity reference mismatch')
        path = root / relative
        stat = path.stat()
        if rows[0]['digest'] != _checked_file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns):
            raise ValueError(f'Migrated HMR parameters changed: {path}')
        return
    if contract.get('geometry_manifest') != 'meta/mano_geometry_manifest.json':
        raise ValueError('Migrated stock MANO requires its persistent HMR manifest')
    manifest = json.loads((root / contract['geometry_manifest']).read_text())
    if manifest['geometry_revision'] != contract['geometry_revision']:
        raise ValueError('MANO manifest revision mismatch')
    path = root / relative
    stat = path.stat()
    if manifest['hmr_sha256'].get(relative) != _checked_file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns):
        raise ValueError(f'Migrated HMR parameters changed: {path}; revalidate parameters and regenerate labels')


def loading_flags(info: dict) -> tuple[bool, bool]:
    """Compatibility flags for legacy wrappers, with installed contracts first."""
    if contract_from_info(info) is not None:
        return True, True
    source_basis = source_basis_from_info(info)
    if source_basis is not None:
        return True, source_basis == "corrected"
    name = str(info.get("dataset_name", "")).lower().removesuffix("_lerobot")
    if not name and info.get("beingh_human", {}).get("mano_layout"):
        raise ValueError("Processed HMR requires a dataset root for source decoding or an installed mano_contract")
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.is_file() else {"datasets": {}}
    decision = registry["datasets"].get(name)
    if decision:
        if decision.get("source_basis") in {"stock", "corrected"} and decision.get("decision_status") in CONFIRMED:
            return True, decision["source_basis"] == "corrected"
        if decision.get("decision_status") in {"no_semantic_left_mano_full_census", "right_model_instances_no_left_basis_change"}:
            return True, True
        raise ValueError(f"Unresolved MANO source basis: {name}")
    return (bool(info.get("mano_flat_hand_mean", name != "gigahands")),
            bool(info.get("mano_left_shapedirs_fix", name in {"interhand26m", "hot3d"})))
