"""Browse a directory of TactoHand inference runs in a local web server."""
from __future__ import annotations

import argparse
import base64
import errno
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, quote, unquote, urlsplit
from zipfile import BadZipFile

import numpy as np

from training.visualization.inference_jobs import PredictionJobs, VIDEO_SUFFIXES

VERTICES = 778


def prediction_field(archive, key: str, dimensions: tuple[int, ...], batch: int) -> np.ndarray:
    values = archive[key]
    if values.ndim == len(dimensions) + 2:
        if not 0 <= batch < len(values):
            raise ValueError(f'Batch index {batch} is outside {key}')
        values = values[batch]
    elif values.ndim != len(dimensions) + 1 or batch != 0:
        raise ValueError(f'Unexpected dimensions or batch index for {key}: {values.shape}')
    if values.shape[1:] != dimensions or not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Invalid values or dimensions in {key}: {values.shape}')
    return values


def encoded(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype)).decode('ascii')


def force_metric(force: np.ndarray, unit: str) -> dict:
    magnitude = np.linalg.norm(force.astype(np.float64), axis=-1)
    return {'data': encoded(magnitude, '<f4'), 'unit': unit,
            'max': max(1e-6, float(np.percentile(magnitude, 98)))}


def load_predictions(path: Path, batch: int, metadata: dict, start: float | None,
                     *, include_vectors: bool = False) -> tuple[dict, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        contact = prediction_field(archive, 'contact_prob', (2, VERTICES), batch)
        distance = prediction_field(archive, 'distance_m', (2, VERTICES), batch)
        physical = 'vertex_force_camera_n_10hz' in archive
        force_key = 'vertex_force_camera_n_10hz' if physical else 'force_normalized_10hz'
        force = prediction_field(archive, force_key, (2, VERTICES, 3), batch)
        if len(distance) != len(contact) or len(force) != len(contact):
            raise ValueError('Prediction fields have different frame counts')
        if np.any((contact < 0) | (contact > 1)) or np.any(distance < 0):
            raise ValueError('Contact must be in [0,1] and decoded distance must be nonnegative')

        timing_key = next((key for key in ('source_frame_times_s', 'target_times_s') if key in archive), None)
        if timing_key is None:
            fps = float(metadata.get('token_hz', 10))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Metadata token_hz must be positive and finite')
            first = metadata.get('start_seconds')
            times = np.arange(len(contact)) / fps + (0. if first is None else float(first))
        else:
            timing_batch = batch if archive[timing_key].ndim == 2 else 0
            times = prediction_field(archive, timing_key, (), timing_batch).astype(np.float64)
        if start is not None:
            times = times - times[0] + start
        if len(times) != len(contact) or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError('Sample times must be finite, increasing and match the predictions')

        indices = None
        if 'source_frame_indices' in archive:
            index_batch = batch if archive['source_frame_indices'].ndim == 2 else 0
            indices = prediction_field(archive, 'source_frame_indices', (), index_batch)
            if len(indices) != len(contact) or np.any(indices < 0) or np.any(indices != np.floor(indices)):
                raise ValueError('Invalid source frame indices')
            indices = indices.astype(np.int64).tolist()
        frame = metadata.get('raw_force_frame', metadata.get('force_frame'))
        local_force = None
        if include_vectors and frame == 'vertex_local' and 'force_normalized_10hz' in archive:
            local_force = (prediction_field(archive, 'force_normalized_10hz', (2, VERTICES, 3), batch)
                           if physical else force)
            if len(local_force) != len(contact):
                raise ValueError('Local force must match the prediction timeline')
    return {
        'times': times.tolist(), 'indices': indices,
        'metrics': {
            'contact': {'data': encoded(contact, '<f4'), 'unit': 'probability', 'max': 1},
            'distance': {'data': encoded(distance * 1000, '<f4'), 'unit': 'mm', 'max': 250},
            'force': force_metric(force, 'N' if physical else 'normalized'),
        },
    }, local_force


def rest_force_vectors(layer, vertices, faces, rotation, force, side: str) -> dict:
    """Pick representative vectors and expand the local components in rest coordinates."""
    import torch
    from core.geometry.force_frames import force_local_to_camera, vertex_local_axes

    topology = (faces, layer.th_J_regressor.cpu().numpy(),
                layer.th_weights.argmax(dim=1).cpu().numpy())
    y, z = vertex_local_axes(vertices, side, topology=topology)
    axes = np.stack((y @ rotation, z @ rotation), axis=-2).astype(np.float32)
    magnitude = np.linalg.norm(force, axis=-1)
    indices = np.column_stack([
        magnitude[:, start:start+12].argmax(axis=1) + start
        for start in range(0, VERTICES, 12)
    ])
    selected = np.take_along_axis(force, indices[..., None], axis=1)
    vectors = force_local_to_camera(torch.from_numpy(selected), torch.from_numpy(axes[indices]))
    return {'indices': encoded(indices, '<u2'), 'values': encoded(vectors.numpy(), '<f4'),
            'normals': encoded(axes[:, 1], '<f4'), 'count': indices.shape[1]}


@lru_cache(maxsize=4)
def mano_template(root: Path, side: str):
    from core.geometry.mano_layer import create_mano_layer

    layer = create_mano_layer(assets_root=root, side=side)
    vertices = layer.th_v_template.detach().cpu().numpy().reshape(-1, 3)
    faces = layer.th_faces.detach().cpu().numpy()
    if (vertices.shape != (VERTICES, 3) or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= VERTICES):
        raise ValueError(f'Invalid {side} MANO template')
    # Use the template's wide surface as the initial view, fingers up.
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    vertical, horizontal = axes[:2]
    if centered[[317, 444, 556, 673]].mean(axis=0) @ vertical < 0:
        vertical = -vertical
    thumb_sign = 1 if side == 'left' else -1
    if (centered[745] @ horizontal) * thumb_sign < 0:
        horizontal = -horizontal
    rotation = np.column_stack((horizontal, vertical, np.cross(horizontal, vertical)))
    display = centered @ rotation
    return layer, vertices, faces, rotation, {
        'vertices': encoded(display, '<f4'), 'faces': encoded(faces, '<u2'),
    }


def mano_templates(root: Path, force: np.ndarray | None = None) -> list[dict]:
    result = []
    for hand, side in enumerate(('left', 'right')):
        layer, vertices, faces, rotation, encoded_mesh = mano_template(root, side)
        mesh = dict(encoded_mesh)
        if force is not None:
            mesh['forceVectors'] = rest_force_vectors(layer, vertices, faces, rotation, force[:, hand], side)
        result.append(mesh)
    return result


def run_prediction(directory: Path) -> Path | None:
    pairs = [path for path in sorted(directory.glob('*.npz'))
             if path.is_file() and path.with_suffix('.json').is_file()]
    named = directory / 'predictions.npz'
    if named in pairs:
        return named
    return pairs[0] if len(pairs) == 1 else None


def run_video(prediction: Path, metadata: dict) -> Path | None:
    for key in ('preview_video', 'input'):
        source = metadata.get(key)
        if not isinstance(source, str) or not source:
            continue
        path = prediction.parent / Path(source).expanduser()
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
            return path
    return None


def read_metadata(prediction: Path) -> dict:
    data = json.loads(prediction.with_suffix('.json').read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('hand_order', ['left', 'right']) != ['left', 'right']:
        raise ValueError('Expected prediction metadata with left/right hand order')
    return data


def byte_range(header: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r'bytes=([0-9]*)-([0-9]*)', header)
    if match is None or not size:
        raise ValueError('Expected one byte range')
    first, last = match.groups()
    if first:
        begin = int(first)
        end = min(size - 1, int(last)) if last else size - 1
    elif last and int(last) > 0:
        begin, end = max(0, size - int(last)), size - 1
    else:
        raise ValueError('Invalid suffix range')
    if begin > end:
        raise ValueError('Unsatisfiable byte range')
    return begin, end


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], outputs: Path, *,
                 mano_root: Path | None = None, batch: int = 0, start: float | None = None,
                 checkpoints: Path | None = None, device: str = 'auto'):
        self.outputs = outputs.resolve()
        self.mano_root = mano_root.expanduser().resolve() if mano_root else None
        self.batch = batch
        self.start = start
        self.html = files('training.visualization').joinpath('inference_viewer.html').read_bytes()
        self.jobs = PredictionJobs(self.outputs, checkpoints.resolve(), device) if checkpoints else None
        super().__init__(address, ViewerHandler)

    def server_close(self) -> None:
        super().server_close()
        if self.jobs:
            self.jobs.close()

    def runs(self) -> list[dict]:
        return [{'id': directory.name} for directory in sorted(self.outputs.iterdir())
                if directory.is_dir() and not directory.is_symlink()
                and not directory.name.startswith('.') and run_prediction(directory) is not None]

    def prediction(self, name: str) -> Path | None:
        if not name or name.startswith('.') or Path(name).name != name:
            return None
        directory = self.outputs / name
        if not directory.is_dir() or directory.is_symlink():
            return None
        return run_prediction(directory)

    def payload(self, name: str, prediction: Path, metadata: dict) -> dict:
        data, force = load_predictions(prediction, self.batch, metadata, self.start,
                                       include_vectors=self.mano_root is not None)
        video = run_video(prediction, metadata)
        source = metadata.get('input')
        title = metadata.get('title')
        data.update(title=title if isinstance(title, str) else Path(source).name if isinstance(source, str) else name,
                    meshes=mano_templates(self.mano_root, force) if self.mano_root else None)
        if force is not None:
            data['vectorScale'] = max(1e-6, float(np.linalg.norm(force, axis=-1).max()))
        if video is not None:
            data['video'] = {'url': f'/api/runs/{quote(name, safe="")}/video'}
        return data


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def log_message(self, format, *args):
        pass

    def reply(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def json(self, status: HTTPStatus, body: dict) -> None:
        self.reply(status, json.dumps(body, allow_nan=False).encode('utf-8'), 'application/json')

    def send_video(self, path: Path) -> None:
        with path.open('rb') as source:
            size = os.fstat(source.fileno()).st_size
            header = self.headers.get('Range')
            status = HTTPStatus.OK
            begin, end = 0, size - 1
            if header is not None:
                try:
                    begin, end = byte_range(header, size)
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(end - begin + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header('Content-Range', f'bytes {begin}-{end}/{size}')
            self.end_headers()
            if self.command != 'HEAD':
                source.seek(begin)
                remaining = end - begin + 1
                while remaining:
                    chunk = source.read(min(remaining, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        try:
            if path in ('/', '/index.html'):
                self.reply(HTTPStatus.OK, self.server.html, 'text/html; charset=utf-8')
                return
            if path == '/api/runs':
                self.json(HTTPStatus.OK, {'runs': self.server.runs()})
                return
            if path == '/api/checkpoints':
                jobs = self.server.jobs
                self.json(HTTPStatus.OK, {'checkpoints': jobs.catalog() if jobs else []})
                return
            job = re.fullmatch(r'/api/jobs/([^/]+)', path)
            if job and self.server.jobs:
                state = self.server.jobs.status(unquote(job[1]))
                if state is not None:
                    self.json(HTTPStatus.OK, state)
                    return
            match = re.fullmatch(r'/api/runs/([^/]+)(/video)?', path)
            if match is not None:
                name = unquote(match[1])
                prediction = self.server.prediction(name)
                if prediction is not None:
                    metadata = read_metadata(prediction)
                    if not match[2]:
                        self.json(HTTPStatus.OK, self.server.payload(name, prediction, metadata))
                        return
                    video = run_video(prediction, metadata)
                    if video is not None:
                        self.send_video(video)
                        return
            self.json(HTTPStatus.NOT_FOUND, {'error': 'Run or video not found'})
        except (BrokenPipeError, ConnectionResetError):
            return
        except (OSError, ValueError, KeyError, ImportError, RuntimeError, EOFError, BadZipFile) as error:
            self.json(HTTPStatus.UNPROCESSABLE_ENTITY, {'error': str(error)})

    do_HEAD = do_GET

    def do_POST(self) -> None:
        request = urlsplit(self.path)
        if request.path != '/api/jobs' or self.server.jobs is None:
            self.json(HTTPStatus.NOT_FOUND, {'error': 'Prediction endpoint not found'})
            return
        if self.headers.get('Content-Type') != 'application/octet-stream':
            self.json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {'error': 'Send the video as application/octet-stream'})
            return
        try:
            query = parse_qs(request.query)
            filename, = query['filename']
            checkpoint, = query['checkpoint']
            length = int(self.headers.get('Content-Length', '0'))
            run = self.server.jobs.submit(filename, checkpoint, self.rfile, length)
            self.json(HTTPStatus.ACCEPTED, {'id': run})
        except (BrokenPipeError, ConnectionResetError):
            return
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.json(HTTPStatus.BAD_REQUEST, {'error': str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputs', nargs='?', type=Path, default=Path('prediction_outputs'),
                        help='directory containing inference runs (default: prediction_outputs/)')
    parser.add_argument('--host', default='127.0.0.1', help='address to bind')
    parser.add_argument('--port', type=int, help='port to bind (default: 8765, or a free port if busy)')
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument('--mano-root', type=Path, default=Path('assets/mano/corrected'),
                          help='MANO asset root (default: assets/mano/corrected)')
    geometry.add_argument('--no-mano', action='store_true', help='display vertex grids')
    parser.add_argument('--checkpoints-root', type=Path, default=Path('checkpoints'),
                        help='checkpoint and YAML directory (default: checkpoints/)')
    parser.add_argument('--device', default='auto', help='prediction device (default: auto)')
    parser.add_argument('--batch-index', type=int, default=0)
    parser.add_argument('--start-seconds', type=float, help='override the first sample time for older exports')
    args = parser.parse_args()
    outputs = args.outputs.expanduser().resolve()
    try:
        outputs.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(str(error))
    mano_root = args.mano_root if not args.no_mano and args.mano_root.expanduser().is_dir() else None
    if (args.port is not None and not 0 <= args.port <= 65535) or args.batch_index < 0:
        parser.error('Use a valid port and a nonnegative batch index')
    if args.start_seconds is not None and (not np.isfinite(args.start_seconds) or args.start_seconds < 0):
        parser.error('--start-seconds must be nonnegative and finite')
    for port in (8765, 0) if args.port is None else (args.port,):
        try:
            server = ViewerServer((args.host, port), outputs, mano_root=mano_root,
                                  batch=args.batch_index, start=args.start_seconds,
                                  checkpoints=args.checkpoints_root.expanduser(), device=args.device)
            break
        except OSError as error:
            if args.port is not None or port == 0 or error.errno != errno.EADDRINUSE:
                parser.error(str(error))
    with server:
        link_host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
        print(f'Prediction runs: {outputs}', flush=True)
        print(f'Open http://{link_host}:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
