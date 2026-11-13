# Usage guide

- [Installation](#installation)
- [Training](#training)
- [Models and data](#model-resources)
- [Inference and evaluation](#prediction)
- [Prediction viewer](#visualization)
- [Building and checking distributions](#distribution)

<a id="installation"></a>

## Installation

Use Python 3.11 with matching PyTorch 2.10 and torchvision 0.25 builds.
Install the dependencies for your workflow from the repository root.

```bash
# Python model API
python -m pip install -e .

# Standalone inference
python -m pip install -e '.[inference]'

# Training and evaluation
python -m pip install -e '.[train]'

# Training and evaluation with TorchCodec video decoding
python -m pip install -e '.[train,video]'

# Optional mesh, visualization, and label-processing dependencies
python -m pip install -e '.[train,visualization,labels]'
```

TorchCodec 0.10 must match your PyTorch, CUDA, and FFmpeg builds. Set
`data.video_backend=cv2` to use OpenCV decoding. The `labels` extra provides
data and mesh utilities. Label solving also requires a Newton/Warp environment.

Geometry-dependent workflows require manotorch and MANO model files. Install
manotorch from its source checkout with
`python -m pip install -e /path/to/manotorch`. Obtain the files from the [MANO project](https://mano.is.tue.mpg.de/)
under its [license](https://mano.is.tue.mpg.de/license.html), then place `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in
`assets/mano/corrected/models/`. The shared loader recognizes the official
files by fingerprint and applies the left-hand shape-basis correction in
memory when required. You can also use an existing corrected pair. The
[asset manifest](../assets/manifest.json) records the expected fingerprints.

| Resource | Setting | Expected input |
|---|---|---|
| Prepared dataset collection | `PROCESSED_ROOT` or `data.processed_root` | Parent directory of the `<dataset>_lerobot` directories |
| Single-dataset root | `DEXYCB_ROOT` or `data.root` | Dataset path in single-root mode |
| Backbone initialization | `PRETRAINED_PATH` or `model.wrench.pretrained_path` | Local backbone weight file |
| MANO models | `MANO_ASSETS_ROOT` | Directory containing `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl` |
| Legacy MANO basis decisions | `MANO_BASIS_DECISIONS` | Reviewed registry for datasets without an installed basis declaration |
| Training statistics and GigaHands allowlist | `core/resources/tactohand/` | Bundled snapshots or explicit overrides in the configuration |

The [resource manifest](../core/resources/tactohand/manifest.json) records
the bundled statistics and allowlist. Preserve the training statistics and
force scale when reproducing a checkpoint. The full MANO mesh contact loss
uses bundled statistics. Legacy reduced-mesh losses require regressors that
match their vertex representation.

<a id="training"></a>

## Training

`tactohand-train` defaults to the 256 px TactoHand experiment and the
`tactohand` data configuration. Inspect the composed configuration before
starting a run.

```bash
tactohand-train --cfg job experiment=tactohand data=tactohand
tactohand-eval --help
tactohand-infer --help
```

To train on prepared DexYCB data with one GPU:

```bash
export PROCESSED_ROOT=/path/to/processed
export PRETRAINED_PATH=/path/to/matching_backbone/model.safetensors
export MANO_ASSETS_ROOT=/path/to/mano
tactohand-train experiment=tactohand data=tactohand \
  'data.datasets=[dexycb]' 'data.val_datasets=[dexycb]' \
  data.batch_size=1 data.num_workers=0 training.devices=1
```

With `pretrained=true` and no local weight path, timm can download backbone
initialization weights. Set `pretrained=false` and `pretrained_path=null`
to test random initialization.

<a id="model-resources"></a>

## Models and data

Use a complete YAML configuration saved with the checkpoint. Copy the run's
`.hydra/config.yaml`, or export the `cfg` section of its `hparams.yaml` to
`model.yaml`. Each prepared dataset's `meta/info.json` identifies its
manifest. See [dataset conventions](datasets.md) and the
[label format](labels_v5.md).

<a id="prediction"></a>

## Inference and evaluation

### Checkpoint configuration

To export a composed configuration, select the experiment and overrides
that produced the checkpoint.

```bash
tactohand-train --cfg job --resolve experiment=tactohand data=tactohand > model.yaml
```

Check the architecture, input size, temporal fusion, `proximity_tau`, and
force frame against the saved run. Inference loads the checkpoint directly,
with backbone initialization disabled. It accepts Lightning `state_dict`
entries under `network.*` or a bare network state dictionary. Missing
parameters and incompatible shapes fail loading.

### Tensor and video inference

```bash
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --input frames.npz --output outputs/prediction.npz
```

`--input` accepts an `.npy` file or an `.npz` file with an `images` array.
Use RGB uint8 with shape `[B,T,3,S,S]` or `[T,3,S,S]`, where `S` matches
`data.image_size`. Floating-point tensors require `--normalized` and must
already follow the training normalization. Use the training clip length and
cadence. The current positional encoding supports up to 128 frames.

The following example checks the input interface with synthetic data.

```python
import numpy as np
from omegaconf import OmegaConf

cfg = OmegaConf.load("model.yaml")
size = int(cfg.data.image_size)
length = int(cfg.data.get("sequence_length", 32))
images = np.random.default_rng(0).integers(
    0, 256, (1, length, 3, size, size), dtype=np.uint8)
np.savez_compressed("frames.npz", images=images)
```

To infer a window from constant-frame-rate video:

```bash
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video clip.mp4 --frames 32 --start-seconds 0 \
  --output outputs/video_prediction.npz
```

Windowed inference selects the closest source frame at each requested
`target_fps` sample, converts it to RGB, and applies the training letterbox.
The video must cover the complete window. Use the full-frame or handcrop
input expected by the model. For full-video inference, use `--full-video`
to sample actual timestamps and merge overlapping windows. For custom
variable-frame-rate windows, sample by timestamp before passing a tensor.

Each successful run writes an NPZ, a same-stem JSON, and a `.done` marker.
Choose a fresh output path for each run. The JSON records paths, shapes,
units, hand order, and preprocessing. Video results include the selected
source frame indices and `target_times_s` for synchronization.

| Field | Shape | Meaning |
|---|---|---|
| `contact_prob` | `[B,T,2,778]` | `sigmoid(gate) * sigmoid(conditional_logit)` |
| `contact_gate_prob` | `[B,T,2]` | Per-hand engagement probability |
| `contact_cond_prob` | `[B,T,2,778]` | Conditional contact probability |
| `proximity_prob` | `[B,T,2,778]` | Proximity score |
| `distance_m` | `[B,T,2,778]` | `-ln(max(p,1e-6))/proximity_tau`, in metres |
| `force_raw_10hz` | `[B,T,2,778,3]` | Raw normalized force head output |
| `force_normalized_10hz` | `[B,T,2,778,3]` | Normalized force after the configured engagement gate, before physical clamping |
| `vertex_force_camera_n_10hz` | `[B,T,2,778,3]` | Camera-frame force in newtons, available when the required geometry and scale are supplied |

The hand axis is ordered left, then right. The current two-hand
vertex-force model emits one image-aligned prediction per input step
(typically 10 Hz). Use the evaluation entry point for legacy
three-substep checkpoints. Standalone inference supports the standard
single-substep model. The JSON sets `force_head_supervised=false` for
contact-only checkpoints.

### Physical force decoding

Supply `--geometry geometry.npz` with the scale used in training and, for
local-frame force heads, the axes for the corresponding input frames.

| Field | Requirement |
|---|---|
| `point_force_scale` | Positive finite scalar or one value per batch item, from `SubstepForceDataModule.point_force_scale` |
| `vertex_local_yz_camera_10hz` | Required for `vertex_force_frame=vertex_local`. Shape `[B,T,2,778,2,3]`, with orthonormal y and z axes in the camera frame |
| `hand_present_10hz` | Optional mask with shape `[B,T,2]`, stored as bool or 0–1 |

The decoder scales the vectors, applies the gate, transforms them to the
camera frame, applies any supplied presence mask, and clamps their physical
magnitude. RGB-only inference returns normalized forces.

### Evaluation

```bash
tactohand-eval --config model.yaml --ckpt /path/to/model.ckpt \
  --processed-root /path/to/processed --datasets dexycb \
  --metrics contact proximity force --device cuda:0 \
  --output outputs/evaluation
```

With `--config`, supply an explicit checkpoint path. Use
`--processed-root` and `--labels-root` to override data paths. Evaluation
requires prepared labels and any geometry assets used by the requested
metrics. Force decoding uses `label_stats` from every dataset in the original
`data.datasets` training mixture, even when evaluating one dataset. Only
the selected evaluation datasets need video decoding and windows. The legacy
`--run ... --ckpt last` entry point records its missing-statistics policy in
the report. Run `tactohand-eval --list-metrics` to list available metrics.

Reports appear under `--output/<timestamp>/tactohand/`. With no
`--output`, standalone evaluation uses `outputs/eval/`.

<a id="metric-mapping"></a>

### Reporting paper metrics

Use `per_dataset[dataset][metric].value` and `.unit`. The convenience
`paper.tables` field currently uses predicted-distance acceleration and
reports force MAE in newtons. For paper results, use the fields below.

| Metric | JSON field | Reporting unit |
|---|---|---|
| Macro-F1 / Micro-F1 | `contact/f1_macro` / `contact/f1_micro` | Multiply by 100 for percentages |
| ADE / ADE-100 | `proximity/ade_mm` / `proximity/ade_clamp100_mm` | mm |
| Distance-error acceleration | `proximity/accel_err_m_s2` | m/s² |
| Vertex / active force MAE | `force/point_mae_n` / `force/active_mae_n` | Multiply by 1000 for mN |
| Force angle | `force/angle_deg` | Degrees |

Use the paper's split, view, input, checkpoint, and metric protocol. The
reported MOW proximity and contact values use single-frame and temporal
variants, respectively.

<a id="visualization"></a>

## Prediction viewer

Run `tactohand-view` and open the URL printed in the terminal. The viewer
prefers port 8765 and chooses a free port when needed. Forward the displayed
port when running through VS Code Remote.

Choose a checkpoint and a video in the browser, or open a completed run.
The viewer discovers checkpoint/YAML pairs in `checkpoints/`, prefers
`tactohand.ckpt`, and writes results to `prediction_outputs/`. Browser
inference requires FFmpeg and ffprobe. It processes the full video with
overlapping windows and uses CUDA when available.

The viewer synchronizes playback with `source_frame_times_s` or
`target_times_s`. Older results can use the JSON sampling rate and
`--start-seconds`. Video paths in `input` and `preview_video` can be
relative to the run directory. Choose a local playback file or set
`preview_video` to a browser-compatible copy when the source codec is
unsupported.

The viewer uses rest-pose meshes from `assets/mano/corrected` by default.
Use `--mano-root` to select another asset directory or `--no-mano` for
vertex grids. The contact view thresholds the interpolated probability at
0.5. The distance view uses a logarithmic scale with 25, 50, 100, and 250 mm
ranges. The force view displays vector magnitude in newtons when camera-frame
forces are available and normalized magnitude otherwise.

Enable **3D force vectors** to display vertex-local force on the rest-pose
meshes. Arrow length is proportional to normalized magnitude with a shared
scale across both hands. Adjust the arrow size or select an arrow to inspect
its vertex and magnitude.

Evaluation rendering with `--visualize N` requires dataset geometry and a
compatible review index. The review path expects `meta/observations.parquet`
and can fail on prepared datasets that use a manifest in its place.

<a id="distribution"></a>

## Building and checking distributions

The [file list](../packaging/public_files.toml) defines the contents of the
source archive and wheel. To build them locally:

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir artifacts/build/wheels .
python setup.py sdist --dist-dir artifacts/build/sdist
python tools/packaging/build_public_source.py --output artifacts/build/source-preview.zip
```

Run the distribution and prediction checks with:

```bash
python tools/checks/check_distribution.py \
  --output artifacts/release_checks/distribution.json \
  --log-dir logs/release_checks/distribution
python tools/checks/check_prediction.py --output artifacts/release_checks/prediction.json
```

The distribution check validates links, builds packages, installs the wheel
outside the checkout, and exercises the command-line interfaces and synthetic
CPU inference. It reuses the active environment's dependencies. The JSON
report records each check, and a `.done` file marks success.
