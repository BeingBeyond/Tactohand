<div align="center">

# TactoHand

### Learning Geometry-Based Dense Hand-Surface Interaction Fields from Video

Ziheng Xi · Weibo Li · Jiayi Yu · Sipeng Zheng · Wanpeng Zhang<br>
Haiweng Xu · Hao Luo · Chuan Mao · Zongqing Lu

[![arXiv](https://img.shields.io/badge/arXiv-2610.39585-b31b1b.svg)](https://arxiv.org/abs/2610.39585)
[![Status](https://img.shields.io/badge/Code-Available-brightgreen.svg)](#release-plan)
[![Project Website](https://img.shields.io/badge/Project-Website-0072B2.svg)](https://zhenqis123.github.io/tactohand-site/)
[![Data & checkpoints](https://img.shields.io/badge/Data%20%26%20checkpoints-Coming%20soon-orange.svg)](#release-plan)

[Release Plan](#release-plan) · [Citation](#citation)

</div>

TactoHand predicts **contact, proximity, and 3D force** at each of the 778 MANO
vertices per hand from monocular RGB video. A visual backbone encodes individual
frames, a temporal transformer combines their features, and task queries decode
the three fields. Training uses geometry-derived contact and proximity labels,
motion-derived force labels, and differentiable physics supervision on eligible
sequences.

[Being-H0.8](https://research.beingbeyond.com/being-h08) uses TactoHand to generate contact and proximity labels from human video for tactile pretraining.

## Getting started

| Workflow | Entry point | Current support |
|---|---|---|
| [Training](#training) | `tactohand-train` | 256px inputs; requires prepared data, backbone weights and MANO |
| [Your own video](#inference-on-your-video) | `tactohand-infer` | RGB clip → numerical interaction fields; physical force decoding needs extra geometry and scale |
| [Model evaluation](#evaluation) | `tactohand-eval` | TactoHand on labeled datasets → metrics and JSON reports |
| [Visualization](#visualization) | `tactohand-view` | Browse prediction runs with synchronized video and per-vertex fields |
| [Dataset usage](#dataset-usage) | Shared manifest and label readers | Episode-level labels, video/geometry access and training batches |

### Installation

Use Python 3.11, PyTorch 2.10 and torchvision 0.25 with a matching CUDA build.
From a checkout containing the source code:

```bash
# Predict interaction fields from your video.
python -m pip install -e '.[inference]'

# Train or evaluate on prepared datasets; TorchCodec also needs system FFmpeg.
python -m pip install -e '.[train,video]'
```

Geometry-dependent workflows additionally require the `manotorch` backend and
MANO model files. Set `MANO_ASSETS_ROOT` to a directory containing
`models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl`; see the
[installation guide](docs/usage.md#installation).
RGB-only inference does not need the datasets or MANO geometry.

### Models and data

TactoHand-256 uses DINOv3 ViT-L and predicts both hands from windows of
32 frames. The training loader chooses clip origins from eligible native video
frames and samples the frames within each clip at 10 Hz. Pretrained
checkpoints and prepared datasets are planned for a separate release.

Use a **complete saved configuration matching the checkpoint**
as `model.yaml`. A Hydra experiment fragment or a newly exported default config
cannot substitute for the run's actual settings. Inference reconstructs the
network from the checkpoint without downloading backbone initialization weights.

## Training

Prepare a parent directory containing `<dataset>_lerobot` children, then set:

```bash
export PROCESSED_ROOT=/path/to/processed
export PRETRAINED_PATH=/path/to/dinov3/model.safetensors
export MANO_ASSETS_ROOT=/path/to/mano

# Inspect the composed configuration before running.
tactohand-train data.batch_size=7 \
  training.devices=8 training.gpu_ids=null training.num_nodes=1 training.strategy=ddp \
  --cfg job --resolve

# Train on eight GPUs on one node.
mkdir -p logs
nohup tactohand-train data.batch_size=7 \
  training.devices=8 training.gpu_ids=null training.num_nodes=1 training.strategy=ddp \
  > logs/train_256.log 2>&1 &
```

The default example uses two gradient-accumulation steps, for an effective
batch size of 112 across eight GPUs. Training logs and checkpoints are written
to the configured output directories. To initialize from task weights while
resetting optimizer and scheduler state, pass
`resume_from=/path/to/model.ckpt resume_weights_only=true`.
Reproducing an existing campaign also requires its saved settings and initialization.
See the [training guide](docs/usage.md#training) for data and runtime details.

## Inference on your video

Supply a constant-frame-rate RGB video and the matching model/config pair:

```bash
mkdir -p prediction_outputs/clip
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video clip.mp4 --start-seconds 0 --frames 32 --device cuda:0 \
  --output prediction_outputs/clip/predictions.npz
```

The command samples the video at the model's input rate and letterboxes each
frame to the configured input size. In this example, 32 samples at 10 Hz span
0–3.1 seconds. The video must cover the requested interval. Set
`--start-seconds` to select another interval, or use `--full-video` in place of
the clip options to process overlapping windows across the full video. Use
`--device cpu` for CPU inference.

Outputs share the requested filename stem:

| File / field | Contents |
|---|---|
| `.npz`: `contact_prob` | Contact probability, `[B,T,2,778]` |
| `.npz`: `proximity_prob`, `distance_m` | Proximity and decoded distance in metres, `[B,T,2,778]` |
| `.npz`: `force_normalized_10hz` | Normalized force vectors, `[B,T,2,778,3]` |
| `.npz`: `source_frame_indices` | Selected source-video frame indices |
| `.npz`: `target_times_s` | Requested sample times in the source video, in seconds |
| `.json` | Field shapes, units, configuration and decoding metadata |
| `.done` | Successful completion marker |

The hand axis lists the left hand first and the right hand second. Use a new
output stem for each run. To pass a tensor instead of a video, use `--input`.

RGB inference returns normalized force vectors. To recover camera-frame forces
in newtons, provide the training normalization scale and, for local-frame
force heads, time-aligned per-vertex axes in `--geometry geometry.npz`.
See [input and force decoding](docs/usage.md#prediction).

## Evaluation

Evaluate **TactoHand** against the prepared reference labels. The default
predictor is our model; external baseline reproduction is separate.

```bash
tactohand-eval --config model.yaml --ckpt model.ckpt \
  --processed-root /path/to/processed --datasets dexycb \
  --split-by-dataset dexycb=test --input-source full \
  --metrics contact proximity force --batch-size 1 --num-workers 0 \
  --device cuda:0 --output outputs/evaluation
```

This evaluates the selected split. For an initial execution check, add
`--limit-batches 1`; omit it for reported results. Reports are written under
`outputs/evaluation/<timestamp>/tactohand/*.json`, including checkpoint identity,
configuration, sample counts, per-dataset metrics and paper-table fields.
For paper comparisons, use the explicit keys and units in `per_dataset`:
the current `paper.tables` acceleration mapping needs correction, and force
MAEs require N-to-mN conversion. See the [metric mapping](docs/usage.md#metric-mapping).

Use the paper's dataset and input protocol explicitly:

| Dataset | Evaluation selection | TactoHand input |
|---|---|---|
| DexYCB | `--datasets dexycb --split-by-dataset dexycb=test` | `--input-source full` |
| HOCap | `--datasets hocap_multiview --split-by-dataset hocap_multiview=test` | `--input-source full` |
| ARCTIC P2 | `--datasets arctic --split-by-dataset arctic=test --views ego_view --subject-filter s05` | `--input-source full` |
| MOW | `--datasets mow --split-by-dataset mow=test` | `--input-source handcrop`; temporal input repeats the static image |

For contact/proximity comparisons, select `--metrics contact proximity`.
The paper reports Macro-F1 over frame–hand pairs with contact in the prediction
or reference, pooled Micro-F1, distance ADE/ADE-100 in mm, and distance-error
acceleration for video. TactoHand contact probabilities are thresholded at 0.5.
The paper uses separate SF/T variants for MOW proximity
and contact; one current temporal checkpoint does not reproduce both columns.

For the force/physics experiment on DexYCB, select `--metrics force rollout`.
Force metrics use valid reference samples aligned with the input images. The
0.4-second rollout (`h4`) starts from the observed state and uses reference
forces between predicted force samples. Read the errors alongside valid counts
and divergence rates. Run `tactohand-eval --list-metrics` for metric names and
units.

Force decoding uses the training mixture's normalization statistics. Select the
evaluation dataset with `--datasets` and keep `data.datasets` in the saved
configuration so the training force scale is preserved. See
[evaluation details](docs/usage.md#prediction) and the paper's evaluation appendix.

## Visualization

### Dataset visualization

The dataset reviewer displays RGB frames, hand and object meshes, and contact,
proximity, and force labels when it can read the dataset layout.

The current `scripts/visualization/visualize_training_batches.py` still reaches
a legacy `meta/observations.parquet` read and fails on the installed manifest-based
dataset layout. Its dataset index integration needs updating before it can serve
as the public dataset viewer. See [visualization status](docs/usage.md#visualization)
for the existing entry points and the remaining work.

### Prediction viewer

Start the viewer from the repository root and open the address printed in the
terminal.

```bash
tactohand-view
```

Choose a checkpoint and a local video in the page, then click **Predict**. The
viewer looks for checkpoint/YAML pairs in `checkpoints/`, selects
`tactohand.ckpt` by default when present, and saves each completed video run
in `prediction_outputs/`. A checkpoint such as `model.ckpt` needs its matching
composed `model.yaml`. Video prediction requires FFmpeg (including ffprobe),
and uses CUDA when available. To browse existing runs, each run directory
needs a prediction NPZ and its same-stem JSON.

Playback and frame selection stay synchronized with the left and right hand
fields. The viewer uses the project demo's binary contact colors (probability
> 0.5), logarithmic distance scale and force magnitude palette.
The viewer uses `assets/mano/corrected` by default for rotatable rest-pose
meshes when MANO assets and the manotorch backend are installed. Use
`--no-mano` to display vertex-index grids.
For predictions in vertex-local coordinates, enable **3D force vectors** to
inspect normalized directions on the rest-pose mesh alongside any surface field.
The viewer reads the source clip from the JSON's `input` path. For HEVC or
another unsupported codec, make a browser-compatible copy and set
`preview_video` in that JSON. Relative video paths are resolved from the run directory.

`--visualize N` on the evaluation command compares predictions with dataset
reference labels. It requires dataset geometry and shares the dataset viewer's
index-format limitation.

## Dataset usage

Point `PROCESSED_ROOT` at the parent of the prepared datasets:

```text
processed/
  dexycb_lerobot/
    meta/info.json       # declares the manifest and label schema
    ...                  # indexed metadata, video, geometry and label shards
  arctic_lerobot/
  ...
```

Use the shared readers instead of assuming physical parquet/video filenames.
For example, read contact and proximity labels from one DexYCB episode:

```python
from pathlib import Path
from core.tactile_labels.decode import load_label_index, read_episode_columns

root = Path("/path/to/processed/dexycb_lerobot")
episode = load_label_index(root, episode_indices=[0])[0]
labels = read_episode_columns(
    root, episode["tactile_groups"],
    ["source_position", "flags", "contact", "signed_distance"],
)
contact = labels["contact"].reshape(-1, 2, 778)
distance_m = labels["signed_distance"].reshape(-1, 2, 778)
```

Use `source_position` and the declared frame mappings to align label rows with
video. Apply the hand-presence and supervision-validity flags before computing
metrics. Compare predicted proximity with the target defined from the signed
reference distance. Contact-only datasets do not provide valid force targets.

For RGB/geometry access, use `core.data.processed_index.ProcessedIndex` and the
shared readers. For model inputs, use `training.data` so sampling, handcrop
intrinsics and force-supervision gates match training.
See the [label schema](docs/labels_v5.md) and [dataset conventions](docs/datasets.md).

## Code organization

| Path | Responsibility |
|---|---|
| `training/tactohand_wrench/` | TactoHand network, losses and model-specific runtime |
| `training/data/`, `training/evaluation/`, `training/visualization/` | Learning data, evaluation protocols and model-result display |
| `core/` | Shared dataset, IO, geometry, physics and label readers |
| `scripts/` | Training, evaluation, data review and other operational entry points |
| `tools/` | Developer checks, packaging and workspace maintenance |

[Usage guide](docs/usage.md) · [Datasets](docs/datasets.md) · [Label format](docs/labels_v5.md)

## Release Plan

- [x] Release the project website.
- [x] Release the source code.
- [ ] Release pretrained checkpoints.
- [ ] Release the dataset.

## License

The code is available under the [MIT License](LICENSE).

## Citation

```bibtex
@article{xi2026tactohand,
  title   = {TactoHand: Learning Geometry-Based Dense Hand-Surface Interaction Fields from Video},
  author  = {Xi, Ziheng and Li, Weibo and Yu, Jiayi and Zheng, Sipeng and
             Zhang, Wanpeng and Xu, Haiweng and Luo, Hao and Mao, Chuan and Lu, Zongqing},
  journal = {arXiv preprint arXiv:2610.39585},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.39585}
}
```
