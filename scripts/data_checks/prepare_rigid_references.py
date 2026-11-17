#!/usr/bin/env python3
"""Recover full multi-rigid-body references offline, with resumable episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import gc
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
_QUERY_DEVICE = 'cpu'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def discover_jobs(processed_root, datasets=None):
    """All splits; never restrict to the old non-target-force eligibility bit."""
    from core.tactile_labels.decode import load_label_index
    roots = ([processed_root / f'{name.removesuffix("_lerobot")}_lerobot'
              for name in datasets] if datasets else sorted(processed_root.glob('*_lerobot')))
    jobs, inventory = [], {}
    for root in roots:
        if not (root / 'meta/info.json').is_file():
            raise FileNotFoundError(f'Missing prepared dataset: {root}')
        rows = load_label_index(root, lazy_groups=True)
        selected = [r for r in rows if r['topology'] == 'rigid' and r['has_force']
                    and len(r['object_ids']) > 1]
        inventory[root.name] = {'episodes': len(selected),
            'solver_rows': sum(int(r['n_rows']) for r in selected),
            'body_rows': sum(int(r['n_rows']) * len(r['object_ids']) for r in selected)}
        for row in selected:
            labels = dict(row)
            for key in ('tactile_groups', 'force_groups'):
                labels[key] = row[key]  # Resolve lazy locators once in parent.
            jobs.append({'root': str(root.resolve()), 'labels': labels})
        print(f'[inventory] {root.name}: {inventory[root.name]}', flush=True)
    jobs.sort(key=lambda j: int(j['labels']['n_rows']) * len(j['labels']['object_ids']), reverse=True)
    return jobs, inventory


def initialize_worker(query_devices=('cpu',), counter=None):
    global _QUERY_DEVICE
    rank = 0
    if counter is not None:
        with counter.get_lock():
            rank = counter.value
            counter.value += 1
    _QUERY_DEVICE = query_devices[rank % len(query_devices)]
    for name in THREAD_ENV:
        os.environ[name] = '1'
    import torch
    import pyarrow as pa
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if _QUERY_DEVICE.startswith('cuda'):
        import warp as wp
        wp.config.quiet = True
        wp.set_device(_QUERY_DEVICE)


def warm_query_kernels(device):
    """Compile once before spawning, so workers reuse the Warp disk cache."""
    import numpy as np
    from core.physics.rigid_multi_contact import nearest_object_owners_batched
    mesh = (np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            np.array([[0, 1, 2]], dtype=np.int32))
    poses = np.eye(4)[None]
    shifted = poses.copy()
    shifted[:, 2, 3] = 2.
    nearest_object_owners_batched(np.array([[[.1, .1, .1]]]), [
        {'mesh': mesh, 'object_to_world': poses, 'dataset': '_warmup', 'object_geometry_id': 'triangle'},
        {'mesh': mesh, 'object_to_world': shifted, 'dataset': '_warmup', 'object_geometry_id': 'triangle'}],
        device=device)


def resolve_query_devices(requested):
    if requested == 'cpu':
        return ('cpu',)
    import torch
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError('GPU ownership requested but no CUDA GPU is visible; refusing CPU fallback')
    if requested == 'cuda':
        return tuple(f'cuda:{i}' for i in range(count))
    if requested.startswith('cuda:') and requested[5:].isdigit() and int(requested[5:]) < count:
        return (requested,)
    raise ValueError(f'Invalid query device: {requested}')


def run_episode(job):
    """Small IPC result; dense arrays remain worker-local and on disk."""
    from core.tactile_labels.rigid_reference_store import read_reference, write_reference
    root, labels = Path(job['root']), job['labels']
    episode = int(labels['episode_index'])
    log = Path(job['log_root']) / root.name / f'episode_{episode:06d}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = 'failed'
    error = None
    with log.open('a', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        # Also redirect native Warp/C++ diagnostics into the episode log.
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            print(f'[start] pid={os.getpid()} episode={episode} query_device={_QUERY_DEVICE}', flush=True)
            try:
                arrays, _ = read_reference(job['output_root'], root, labels)
            except (ValueError, OSError, KeyError):
                arrays = None  # Stale/corrupt artifacts are rebuilt atomically.
            if arrays is not None:
                status = 'cached'
                del arrays
            else:
                from core.tactile_labels.rigid_reference_prepare import prepare_episode
                prepare_episode(root, labels, Path(job['output_root']),
                    mano_chunk=job['mano_chunk'], query_device=_QUERY_DEVICE,
                    trust_degenerate=job.get('trust_degenerate', True))
                status = 'ready'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
            write_reference(job['output_root'], root, labels, error=error)
        finally:
            print(f'[finish] status={status} seconds={time.monotonic() - started:.3f}', flush=True)
            for target, fd in zip((1, 2), saved):
                os.dup2(fd, target)
                os.close(fd)
    gc.collect()
    return {'dataset': root.name, 'episode': episode, 'status': status,
            'seconds': time.monotonic() - started, 'error': error, 'log': str(log)}


def run_jobs(jobs, output_root, workers, *, executor_factory=ProcessPoolExecutor,
             worker=run_episode, query_devices=('cpu',)):
    """Bounded submission, incremental reports, nonzero result on any failure."""
    started = time.monotonic()
    report = {'total': len(jobs), 'completed': 0, 'ready': 0, 'cached': 0,
              'failed': 0, 'workers': workers, 'query_devices': list(query_devices),
              'trust_degenerate': bool(jobs[0].get('trust_degenerate', True)) if jobs else None,
              'results': []}
    def save():
        report['seconds'] = time.monotonic() - started
        atomic_json(output_root / 'summary.json', report)

    save()
    options = dict(max_workers=workers)
    if executor_factory is ProcessPoolExecutor:
        context = multiprocessing.get_context('spawn')
        options.update(mp_context=context, initializer=initialize_worker,
                       initargs=(query_devices, context.Value('i', 0)))
    try:
        with executor_factory(**options) as pool:
            iterator = iter(jobs)
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < workers * 2:
                    job = next(iterator, None)
                    if job is None:
                        exhausted = True
                    else:
                        pending[pool.submit(worker, job)] = job
                if not pending:
                    break
                finished, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'dataset': Path(job['root']).name,
                            'episode': int(job['labels']['episode_index']),
                            'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                    report['results'].append(result)
                    report['completed'] += 1
                    report[result['status']] += 1
                save()
                print(f'[progress] {report["completed"]}/{len(jobs)} '
                      f'ready={report["ready"]} cached={report["cached"]} '
                      f'failed={report["failed"]} elapsed={report["seconds"]:.0f}s', flush=True)
    except Exception as exc:
        report['fatal_error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    success = report['failed'] == 0 and report['completed'] == report['total']
    if success:
        atomic_json(output_root / 'preparation.done', {
            'total': report['total'], 'seconds': report['seconds'], 'force_labels_unchanged': True})
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--processed-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--log-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=80)
    parser.add_argument('--mano-chunk', type=int, default=128)
    parser.add_argument('--query-device', default='cuda', help='cuda: use all visible GPUs; cuda:N; cpu for debugging')
    parser.add_argument('--dataset', action='append', help='Optional dataset filter; default all prepared datasets/splits')
    parser.add_argument('--plan-only', action='store_true', help='Inventory metadata only; no geometry preparation')
    parser.add_argument('--check', action='store_true', help='Check imports/arguments; do not read datasets or start workers')
    parser.add_argument('--exact-degenerate', dest='trust_degenerate', action='store_false',
                        help='Use float64 exact ownership for meshes with thin triangles (slow)')
    args = parser.parse_args()
    if args.workers < 1 or args.mano_chunk < 1:
        parser.error('workers and mano-chunk must be positive')
    for name in THREAD_ENV:
        os.environ[name] = '1'
    if args.check:
        from core.tactile_labels.rigid_reference_prepare import prepare_episode
        import warp
        print(f'OK: workers={args.workers} query_device={args.query_device} '
              f'trust_degenerate={args.trust_degenerate} scope=all-multi-rigid-splits '
              f'output={args.output_root}; no data preparation started')
        return 0
    args.output_root = args.output_root.resolve()
    args.log_root = args.log_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs, inventory = discover_jobs(args.processed_root, args.dataset)
        atomic_json(args.output_root / 'inventory.json', {'scope': 'all splits, all force-labeled rigid bodies',
            'datasets': inventory, 'episodes': len(jobs), 'processed_root': str(args.processed_root.resolve())})
        if not jobs:
            raise ValueError('No multi-rigid-body episodes found; refusing an empty success marker')
        if args.plan_only:
            return 0
        devices = resolve_query_devices(args.query_device)
        (args.output_root / 'preparation.done').unlink(missing_ok=True)
        for job in jobs:
            job.update(output_root=str(args.output_root), log_root=str(args.log_root),
                       mano_chunk=args.mano_chunk, trust_degenerate=args.trust_degenerate)
        print(f'[prepare] episodes={len(jobs)} workers={args.workers} '
              f'query_devices={devices} trust_degenerate={args.trust_degenerate} '
              f'resume=verified-payloads force-resolve=no', flush=True)
        warm_query_kernels(devices[0])
        return run_jobs(jobs, args.output_root, args.workers, query_devices=devices)


if __name__ == '__main__':
    raise SystemExit(main())
