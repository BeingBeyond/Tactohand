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
MANO model files. Place the files under `assets/mano/corrected/models/`, or
set `MANO_ASSETS_ROOT` to another asset root. See the
[installation guide](docs/usage.md#installation).
RGB-only inference does not need the datasets or MANO geometry.

### Models and data

TactoHand-256 uses DINOv3 ViT-L and predicts both hands from windows of
32 frames. The training loader chooses clip origins from eligible native video
frames and samples the frames within each clip at 10 Hz. Pretrained
checkpoints and prepared datasets are planned for a separate release.

Pair each checkpoint with its [saved configuration](docs/usage.md#model-resources).

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

Each run writes predictions, JSON metadata, and a completion marker. See the
[output fields and force decoding guide](docs/usage.md#prediction) for shapes,
units, tensor inputs, and camera-frame forces.

## Evaluation

Evaluate a checkpoint against prepared reference labels:

```bash
tactohand-eval --config model.yaml --ckpt model.ckpt \
  --processed-root /path/to/processed --datasets dexycb \
  --split-by-dataset dexycb=test --input-source full \
  --metrics contact proximity force --batch-size 1 --num-workers 0 \
  --device cuda:0 --output outputs/evaluation
```

Add `--limit-batches 1` for a smoke test. Reports appear under
`outputs/evaluation/<timestamp>/tactohand/`. Use the
[metric mapping](docs/usage.md#metric-mapping) for paper results.

Use the paper's dataset and input protocol explicitly:

| Dataset | Evaluation selection | TactoHand input |
|---|---|---|
| DexYCB | `--datasets dexycb --split-by-dataset dexycb=test` | `--input-source full` |
| HOCap | `--datasets hocap_multiview --split-by-dataset hocap_multiview=test` | `--input-source full` |
| ARCTIC P2 | `--datasets arctic --split-by-dataset arctic=test --views ego_view --subject-filter s05` | `--input-source full` |
| MOW | `--datasets mow --split-by-dataset mow=test` | `--input-source handcrop`; temporal input repeats the static image |

Use `--metrics contact proximity` for surface fields and
`--metrics force rollout` for the DexYCB physics evaluation. The 0.4-second
rollout starts from the observed state and uses reference forces between
predicted force samples.

## Visualization

```bash
tactohand-view
```

Open the printed URL to predict from a local video or browse a completed
run. See the [viewer guide](docs/usage.md#visualization) for checkpoint
locations, playback, and 3D forces.

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

See the [label format and Python example](docs/labels_v5.md#reading-an-episode)
and [dataset conventions](docs/datasets.md).

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
