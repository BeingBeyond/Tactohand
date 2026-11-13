#!/usr/bin/env python3
"""Switch prepared datasets to an audited native timeline release."""
from argparse import ArgumentParser
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.data.processed_index import ProcessedIndex


def write_atomic(path, value):
    temporary = path.with_name(f'.{path.name}.native_release_pending')
    temporary.write_bytes(value)
    temporary.replace(path)


def activation_plan(root, release, audit):
    plan = []
    for name, evidence in sorted(audit.items()):
        dataset = root / f'{name}_lerobot'
        directory = dataset / f'meta/index/native_timeline/{release}'
        candidate = json.loads((directory / 'candidate.json').read_text())
        receipt = json.loads((directory / 'sealed.json').read_text())
        manifest_path = receipt['manifest']
        index = ProcessedIndex(dataset, manifest_path=manifest_path)
        if (hashlib.sha256(index.path(manifest_path).read_bytes()).hexdigest()
                != receipt['manifest_sha256']
                or index.revision != evidence['revision']
                or evidence['episodes'] != candidate['episodes']
                or evidence['sample_shards'] != candidate['sample_shards']
                or index.has_modality('tactile.tokens')
                or not index.has_modality('tactile.timeline')
                or index.manifest['contracts']['beingh']['labels']['timeline'] != 'solver_rows_v1'):
            raise ValueError(f'{name}: audit does not match the sealed release')
        path = dataset / 'meta/info.json'
        original = path.read_bytes()
        info = json.loads(original)
        if info['manifest'] == index.manifest_path and info['revision'] == index.revision:
            continue
        if (info['manifest'] != candidate['source_manifest']
                or info['revision'] != candidate['source_revision']
                or hashlib.sha256(index.path(info['manifest']).read_bytes()).hexdigest()
                != candidate['source_manifest_sha256']):
            raise ValueError(f'{name}: active source differs from the staged source')
        info.update(manifest=index.manifest_path, revision=index.revision)
        updated = (json.dumps(info, indent=2) + '\n').encode()
        plan.append((name, path, original, updated))
    return plan


def activate(plan, backup):
    backup.mkdir(parents=True, exist_ok=True)
    for name, _, original, _ in plan:
        path = backup / f'{name}.info.json'
        if not path.exists():
            write_atomic(path, original)
        elif path.read_bytes() != original:
            raise ValueError(f'Conflicting rollback pointer: {path}')
    written = []
    try:
        for name, path, original, updated in plan:
            if path.read_bytes() != original:
                raise ValueError(f'{name}: active pointer changed during activation')
            write_atomic(path, updated)
            written.append((path, original, updated))
        for path, _, updated in written:
            if path.read_bytes() != updated:
                raise ValueError(f'Activation readback failed: {path}')
    except BaseException:
        for path, original, updated in reversed(written):
            if path.read_bytes() == updated:
                write_atomic(path, original)
        raise


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--processed-root', type=Path, required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', args.release) or not audit
            or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in audit)):
        parser.error('provide a named release and a nonempty audit of dataset names')
    if not args.apply:
        plan = activation_plan(args.processed_root, args.release, audit)
    else:
        with (args.processed_root / '.native_timeline_activation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = activation_plan(args.processed_root, args.release, audit)
            activate(plan, args.backup)
    print(f'Validated {len(audit)} datasets; '
          f'{len(plan)} pointers {"updated" if args.apply else "ready to update"}.')


if __name__ == '__main__':
    main()
