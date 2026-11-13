"""Predict from RGB tensors or video using an explicit config and checkpoint."""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from training.tactohand_wrench.runtime.checkpoints import load_network, load_prediction_config
from training.tactohand_wrench.runtime.inference import decode_predictions, validate_images, validate_inference_config


def read_video(path: Path, *, frames: int, start_seconds: float,
               target_fps: float, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    from training.data.preprocessing import letterbox_square

    if not 1 <= frames <= 128 or not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("--frames must be in [1,128] and --start-seconds non-negative and finite")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("data.target_fps must be positive and finite")
    capture = cv2.VideoCapture(str(path.expanduser().resolve()))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video must declare a positive CFR rate; use explicit tensor inputs for VFR sources")
        indices = np.rint((start_seconds + np.arange(frames) / target_fps) * fps).astype(np.int64)
        images = []
        ordinal = 0
        current = None
        for index in indices:
            while ordinal <= index:
                ok, current = capture.read()
                if not ok:
                    raise ValueError(f"Video ends before requested source frame {index}; choose a shorter clip")
                ordinal += 1
            rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
            images.append(letterbox_square(rgb, image_size).transpose(2, 0, 1))
        return np.stack(images)[None], indices
    finally:
        capture.release()


def full_video_timing(path: Path, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Map a uniform prediction clock to nearest presentation timestamps."""
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("data.target_fps must be positive and finite")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries", "frame=best_effort_timestamp_time:stream=duration", "-of", "json", str(path),
    ], check=True, capture_output=True)
    info = json.loads(probe.stdout)
    timestamps = np.array([float(row["best_effort_timestamp_time"]) for row in info.get("frames", [])])
    if not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError("Video needs finite presentation timestamps in display order")
    timestamps -= timestamps[0]
    gaps = np.diff(timestamps)
    spacing = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 1 / target_fps
    duration = timestamps[-1] + spacing
    try:
        reported = float(info["streams"][0]["duration"])
    except (IndexError, KeyError, TypeError, ValueError):
        reported = duration
    if timestamps[-1] < reported <= duration + spacing:
        duration = reported
    times = np.arange(max(1, math.ceil(duration * target_fps - 1e-8))) / target_fps
    right = np.minimum(np.searchsorted(timestamps, times), len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(times - timestamps[left]) <= abs(timestamps[right] - times), left, right)
    return indices, times


def indexed_video_frames(path: Path, indices: np.ndarray, image_size: int):
    import cv2
    from training.data.preprocessing import letterbox_square

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        ordinal = -1
        frame = None
        for index in indices:
            if index != ordinal:
                while ordinal < index:
                    ok, current = capture.read()
                    if not ok:
                        raise ValueError(f"Video decoder stopped before source frame {index}")
                    ordinal += 1
                rgb = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
                frame = letterbox_square(rgb, image_size).transpose(2, 0, 1)
            yield frame
    finally:
        capture.release()


def predict_full_video(path, network, cfg, device):
    indices, times = full_video_timing(path, float(cfg.data.get("target_fps", 10.)))
    length = min(len(indices), int(cfg.data.get("sequence_length", 32)), network.temporal.position.shape[1])
    if length < 1:
        raise ValueError("The model must support at least one frame")
    tail = len(indices) - length
    starts = list(range(0, tail + 1, max(1, length // 2)))
    if starts[-1] != tail:
        starts.append(tail)
    print(f"Window 0/{len(starts)}", flush=True)
    margin = np.minimum(np.arange(length), np.arange(length)[::-1])
    coverage = np.full(len(indices), -1, dtype=np.int32)
    arrays = {}
    frames = indexed_video_frames(path, indices, int(cfg.data.image_size))
    buffer = deque(maxlen=length)
    consumed = 0
    try:
        with torch.inference_mode():
            for index, start in enumerate(starts):
                while consumed < start + length:
                    buffer.append(next(frames))
                    consumed += 1
                tensor = validate_images(np.stack(buffer), cfg).to(device)
                predictions = decode_predictions(network(tensor), cfg)
                region = slice(start, start + length)
                replace = margin > coverage[region]
                coverage[region][replace] = margin[replace]
                for key, value in predictions.items():
                    values = value.cpu().numpy()
                    if key not in arrays:
                        arrays[key] = np.empty((1, len(indices), *values.shape[2:]), dtype=values.dtype)
                    arrays[key][:, region][:, replace] = values[:, replace]
                print(f"Window {index + 1}/{len(starts)}", flush=True)
    finally:
        frames.close()
    return arrays, indices, times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help=".npy tensor or .npz with images; RGB uint8 [B,T,3,S,S]")
    source.add_argument("--video", type=Path, help="one CFR video clip; shared letterbox preprocessing")
    parser.add_argument("--frames", type=int, help="video token count (default: config sequence_length)")
    parser.add_argument("--full-video", action="store_true", help="predict the complete video using overlapping windows")
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--normalized", action="store_true", help="tensor input already contains training-normalized floats")
    parser.add_argument("--geometry", type=Path, help="optional NPZ with force scale and local camera axes")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True, help="new .npz output, plus same-stem JSON metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.full_video and (args.video is None or args.frames is not None or args.start_seconds != 0 or args.geometry):
        raise ValueError("--full-video requires --video and no clip bounds or geometry")
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cfg = load_prediction_config(args.config)
    validate_inference_config(cfg)
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    metadata_path = output.with_suffix(".json")
    done = output.with_suffix(".done")
    if any(path.exists() for path in (output, metadata_path, done)):
        raise FileExistsError(f"Choose a new output filename: {output}")
    source_indices = None
    target_times = None
    if args.video is not None:
        if args.normalized:
            raise ValueError("--normalized applies only to tensor input")
        if not args.full_video:
            images, source_indices = read_video(
                args.video, frames=(args.frames if args.frames is not None else int(cfg.data.get("sequence_length", 32))),
                start_seconds=args.start_seconds, target_fps=float(cfg.data.get("target_fps", 10.)),
                image_size=int(cfg.data.image_size))
    else:
        if args.frames is not None or args.start_seconds != 0:
            raise ValueError("--frames/--start-seconds apply only to --video")
        loaded = np.load(args.input.expanduser(), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                images = loaded["images"]
        else:
            images = loaded
    geometry = None
    if args.geometry is not None:
        with np.load(args.geometry.expanduser(), allow_pickle=False) as archive:
            geometry = {key: archive[key] for key in archive.files}
    network = load_network(cfg, args.ckpt, device)
    if args.full_video:
        arrays, source_indices, target_times = predict_full_video(args.video, network, cfg, device)
    else:
        tensor = validate_images(images, cfg, args.normalized)
        if tensor.shape[1] > network.temporal.position.shape[1]:
            raise ValueError("Input clip exceeds the checkpoint's maximum temporal length")
        with torch.inference_mode():
            values = decode_predictions(network(tensor.to(device)), cfg, geometry)
        arrays = {key: value.cpu().numpy() for key, value in values.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Prediction contains non-finite values; no output written")
    if source_indices is not None:
        arrays["source_frame_indices"] = source_indices
        arrays["target_times_s"] = target_times if target_times is not None else (
            args.start_seconds + np.arange(len(source_indices)) / float(cfg.data.get("target_fps", 10.)))
    force_cfg = cfg.loss.get("wrench", {})
    metadata = {
        "schema_version": 1,
        "config_path": str(args.config.expanduser().resolve()),
        "checkpoint": str(args.ckpt.expanduser().resolve()),
        "input": str((args.video or args.input).expanduser().resolve()),
        "start_seconds": args.start_seconds if args.video else None,
        "token_hz": float(cfg.data.get("target_fps", 10.)),
        "image_size": int(cfg.data.image_size),
        "device": device,
        "full_video": args.full_video,
        "input_preprocessing": ("RGB letterbox to image_size; ImageNet normalization in network" if args.video
                                else "caller-normalized floats" if args.normalized
                                else "caller-provided RGB uint8 square tensors; ImageNet normalization in network"),
        "proximity_tau_per_m": float(cfg.data.get("proximity_tau", 10.)),
        "sample_alignment": "image-aligned substep 0 only",
        "hand_order": ["left", "right"],
        "raw_force_frame": str(cfg.data.get("vertex_force_frame", "camera")),
        "force_head_supervised": any(float(force_cfg.get(key, 0)) > 0 for key in (
            "point_force_weight", "wrench_force_weight", "wrench_torque_weight")),
        "geometry": str(args.geometry.expanduser().resolve()) if args.geometry else None,
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "units": {"contact_prob": "1", "contact_gate_prob": "1", "contact_cond_prob": "1", "proximity_prob": "1",
                  "distance_m": "m", "force_raw_10hz": "normalized", "force_normalized_10hz": "normalized",
                  **({"hand_visibility_prob_10hz": "1"} if "hand_visibility_prob_10hz" in arrays else {}),
                  **({"vertex_force_camera_n_10hz": "N"} if geometry is not None else {})},
        "physical_force_available": geometry is not None,
        "rollout_available": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    done.touch(exist_ok=False)
    print(f"Predictions: {output}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
