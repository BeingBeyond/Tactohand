"""Queue local RGB video predictions for the TactoHand run viewer."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from threading import Lock
from typing import BinaryIO
from uuid import uuid4

VIDEO_SUFFIXES = {'.mp4', '.webm', '.ogg', '.ogv', '.mov', '.m4v', '.avi', '.mkv'}


class PredictionJobs:
    def __init__(self, outputs: Path, checkpoints: Path, device: str):
        self.outputs = outputs
        self.checkpoints = checkpoints
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tactohand-infer')
        self.lock = Lock()
        self.jobs: dict[str, dict[str, str]] = {}

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)

    def checkpoint(self, name: str) -> tuple[Path, Path] | None:
        if not name or name.startswith('.') or Path(name).name != name:
            return None
        weights = self.checkpoints / name
        if not weights.is_file() or weights.suffix.lower() not in {'.ckpt', '.pt', '.pth'}:
            return None
        for suffix in ('.yaml', '.yml'):
            config = weights.with_suffix(suffix)
            if config.is_file():
                return weights, config
        return None

    def catalog(self) -> list[dict[str, str]]:
        if not self.checkpoints.is_dir():
            return []
        names = [path.name for path in self.checkpoints.iterdir() if self.checkpoint(path.name)]
        return [{'id': name} for name in sorted(names, key=lambda name: (name != 'tactohand.ckpt', name))]

    def status(self, run: str) -> dict[str, object] | None:
        with self.lock:
            job = self.jobs.get(run)
            if job is None:
                return None
            state: dict[str, object] = dict(job)
        if state.get('phase') == 'inference':
            log = self.outputs / f'.pending-{run}' / 'inference.log'
            try:
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 8192))
                    matches = re.findall(rb'(?m)^Window ([0-9]+)/([0-9]+)\r?\n', stream.read())
            except OSError:
                matches = []
            if matches:
                completed, total = map(int, matches[-1])
                if total > 0 and 0 <= completed <= total:
                    state['progress'] = {'completed': completed, 'total': total}
                    state['detail'] = ('Saving predictions…' if completed == total else
                                       f'Running video inference · {completed}/{total} windows')
        return state

    def update(self, run: str, status: str, detail: str, *, phase: str = '') -> None:
        with self.lock:
            self.jobs[run] = {'id': run, 'status': status, 'detail': detail, 'phase': phase}

    def submit(self, name: str, checkpoint: str, stream: BinaryIO, length: int) -> str:
        if length <= 0:
            raise ValueError('The video upload is empty')
        weights_config = self.checkpoint(checkpoint)
        if weights_config is None:
            raise ValueError('Choose a checkpoint with a matching YAML config')
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError('Choose a supported video file')
        stem = re.sub(r'[^A-Za-z0-9_-]+', '-', Path(name).stem).strip('-_')[:64] or 'video'
        run = f'{datetime.now():%Y%m%d_%H%M%S}_{stem}_{uuid4().hex[:8]}'
        stage = self.outputs / f'.pending-{run}'
        stage.mkdir()
        source = stage / f'source{suffix}'
        try:
            with source.open('xb') as destination:
                while length:
                    chunk = stream.read(min(length, 1024 * 1024))
                    if not chunk:
                        raise ValueError('The upload ended before the declared file size')
                    destination.write(chunk)
                    length -= len(chunk)
        except BaseException:
            source.unlink(missing_ok=True)
            stage.rmdir()
            raise
        self.update(run, 'queued', 'Waiting for the prediction worker')
        self.executor.submit(self.predict, run, name, stage, source, *weights_config)
        return run

    def predict(self, run: str, name: str, stage: Path, source: Path, weights: Path, config: Path) -> None:
        log = stage / 'inference.log'
        output = stage / 'predictions.npz'
        self.update(run, 'running', 'Loading checkpoint and sampling video…', phase='inference')
        try:
            with log.open('wb') as stream:
                environment = os.environ.copy()
                environment.setdefault('OMP_NUM_THREADS', '2')
                environment.setdefault('MKL_NUM_THREADS', '2')
                subprocess.run([
                    sys.executable, '-u', '-m', 'training.infer_tactohand',
                    '--config', str(config), '--ckpt', str(weights),
                    '--video', str(source), '--full-video', '--device', self.device,
                    '--output', str(output),
                ], stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)
                if not output.with_suffix('.done').is_file():
                    raise ValueError('Inference did not produce a completion marker')
                metadata_path = output.with_suffix('.json')
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                metadata.update(input=source.name, title=name)
                if source.suffix != '.webm':
                    self.update(run, 'running', 'Inference finished · preparing browser video…', phase='preview')
                    preview = stage / 'preview.webm'
                    try:
                        subprocess.run([
                            'ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                            '-map', '0:v:0', '-map', '0:a:0?', '-vsync', '0',
                            '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv420p', '-b:v', '0', '-crf', '34',
                            '-deadline', 'realtime', '-cpu-used', '5', '-threads', '2',
                            '-c:a', 'libopus', str(preview),
                        ], stdout=stream, stderr=subprocess.STDOUT, check=True)
                    except (OSError, subprocess.CalledProcessError) as error:
                        preview.unlink(missing_ok=True)
                        stream.write(f'Preview conversion failed: {error}\n'.encode())
                    else:
                        metadata['preview_video'] = preview.name
                metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
            stage.rename(self.outputs / run)
            self.update(run, 'complete', 'Prediction saved')
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            detail = str(error)
            if isinstance(error, subprocess.CalledProcessError) and error.returncode == -signal.SIGKILL:
                detail = 'The prediction worker was killed (SIGKILL). Check available memory or run on a GPU host.'
            elif log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 4096))
                    lines = stream.read().decode(errors='replace').splitlines()
                if lines:
                    detail = lines[-1]
            self.update(run, 'failed', detail)
