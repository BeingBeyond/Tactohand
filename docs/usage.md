# Usage

[English](usage.md) | [简体中文](usage.zh-CN.md)

<a id="setup"></a>

## Installation

Use Python 3.11, PyTorch 2.10 and torchvision 0.25 with a matching CUDA build.
Choose the dependencies for your workflow:

```bash
python -m pip install '.[inference]'
python -m pip install -e '.[train,video]'
```

Video inference needs FFmpeg and ffprobe. The `video` extra installs TorchCodec
0.10; it must match PyTorch, CUDA and FFmpeg. Use `data.video_backend=cv2` for
OpenCV decoding. Full label solving needs a separately prepared Newton/Warp
environment; the `labels` extra supplies only data and mesh utilities.

| Resource | Configuration |
|---|---|
| Prepared datasets | `PROCESSED_ROOT` or `data.processed_root`: parent of `<dataset>_lerobot` directories |
| Backbone initialization | `PRETRAINED_PATH` or `model.wrench.pretrained_path`: local weights matching the backbone |
| MANO models | `MANO_ASSETS_ROOT`: directory containing `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl` |

For geometry-dependent workflows, install
[manotorch](https://github.com/lixiny/manotorch) from its upstream repository:

```bash
git clone https://github.com/lixiny/manotorch.git
python -m pip install -e ./manotorch
```

The original MANO files used here have a known
[left-hand shape-basis issue](https://github.com/vchoutas/smplx/issues/48): the
x component of the left-hand shape blend shapes (`shapedirs`) is not mirrored,
so identical shape parameters can produce inconsistent left and right hand shapes.
We flip this component to preserve consistent hand shape across both hands.

`assets/mano/stock` holds the original files; `assets/mano/corrected` holds the
corrected files. The loader uses file fingerprints to apply the correction only
when needed; no manual PKL editing is required. Existing fitted parameters must
use their matching basis: changing the model files does not convert those parameters.

## Prediction viewer

Open a video in your browser to explore predicted contact, distance and force,
with visibility shown above each hand.

Try the [online demo](https://huggingface.co/spaces/zhenqis123/TactoHand) with videos up to 30 seconds and 100 MB,
or follow the steps below to run locally. The demo places video beside interactive
hand meshes, with synchronized playback and per-hand visibility. Choose one of
the eight example clips or upload your own video. Drag the meshes to rotate them;
GPU requests may queue.

### Start the viewer

Download `tactohand.ckpt` and its matching `tactohand.yaml` from either source:

- [Hugging Face](https://huggingface.co/zhenqis123/TactoHand/tree/main)
- [Baidu Netdisk](https://pan.baidu.com/s/1iG-hTmECh_FrkJKIgWdSiQ?pwd=898f) (access code: `898f`; no expiry)

Place both files in `checkpoints/`, then start the viewer:

```bash
mkdir -p checkpoints
cp /path/to/downloads/tactohand.ckpt checkpoints/
cp /path/to/downloads/tactohand.yaml checkpoints/
tactohand-view --port 8765
```

Open [http://localhost:8765](http://localhost:8765). In **Server**, choose the
checkpoint and device, then click **Apply settings**. Under **New prediction**,
upload a video and click **Predict**. Play the video or drag the timeline to
inspect the results; use **Prediction run** to reopen a saved result.

**Client** only selects the server address. **Server** manages the checkpoint,
device and **Accept new predictions** switch. Turning this switch off pauses new
requests; running and queued jobs continue with their original settings.

To use MANO models from another location:

```bash
tactohand-view --port 8765 --mano-root /path/to/mano
```

<a id="remote-inference"></a>

### Use a remote GPU

On the GPU machine, prepare the checkpoint and YAML as above, then start the
service from its checkout:

```bash
mkdir -p logs
nohup python -u -m training.visualization.inference_viewer prediction_outputs \
  --api-only --host 127.0.0.1 --port 8766 \
  --checkpoints-root checkpoints --mano-root /path/to/mano --device cuda:0 \
  > logs/inference_api.log 2>&1 < /dev/null &
```

On your local machine, replace `USER@GPU_HOST` with your SSH login and leave
this terminal open:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18766:127.0.0.1:8766 USER@GPU_HOST
```

In another local terminal, start the dashboard from your checkout:

```bash
python3 -m training.visualization.inference_dashboard --port 8765
```

Open [http://localhost:8765](http://localhost:8765). In **Client**, enter
`http://127.0.0.1:18766` and click **Connect**. Manage the GPU server's checkpoint
and device in **Server**, then upload your video under **New prediction**.
Keep the SSH tunnel open while using the dashboard.

## Training

Prepare [datasets](datasets.md), backbone weights and MANO. The default
`experiment=tactohand_256 data=tactohand` is the formal 256px, 1,000-epoch
configuration. `experiment=tactohand_256_fresh` provides the 500-epoch fresh
initialization configuration. The 1,000-epoch configuration accepts optional
weights-only initialization through `RESUME_FROM`.

Download `training_resources/tactohand_v1.tar.gz` from the same dataset release
and extract it using this layout:

```text
<collection>/data/<dataset>_lerobot/
<collection>/training_resources/tactohand_v1/
  statistics/{scales.json,output_scales.json,contact.npz,force.npz}
  rigid_references/<dataset>_lerobot/episode_*/{manifest.json,bodies.parquet}
```

```bash
mkdir -p /path/to/collection/training_resources
tar -xzf tactohand_v1.tar.gz -C /path/to/collection/training_resources
export PROCESSED_ROOT=/path/to/collection/data
export TACTOHAND_TRAINING_RESOURCES=/path/to/collection/training_resources/tactohand_v1
export PRETRAINED_PATH=/path/to/backbone/model.safetensors
export MANO_ASSETS_ROOT=/path/to/mano
python scripts/data_checks/check_training_resources.py \
  --processed-root "$PROCESSED_ROOT" --resources-root "$TACTOHAND_TRAINING_RESOURCES"
tactohand-train --cfg job --resolve
mkdir -p logs
nohup env GPUS=8 bash scripts/training/recipes/tactohand/train_256.sh \
  > logs/train.log 2>&1 < /dev/null &
```

The repository contains sequence splits and the object allowlist. Statistics
and rigid references are distributed with the dataset and bind its exact
manifest hashes. `output_scales.json` fixes the trained model output normalization;
`scales.json` contains the label-fitted normalization for fresh initialization.
The default uses 256px inputs and batch size 7 per GPU.
Another dataset publication needs matching bindings, even
when numerical labels are unchanged. Use a fresh output directory when
rebuilding resources:

```bash
python scripts/data_checks/build_training_statistics.py \
  --processed-root "$PROCESSED_ROOT" --output /path/to/new_resources/statistics
python scripts/data_checks/prepare_rigid_references.py \
  --processed-root "$PROCESSED_ROOT" --output-root /path/to/new_resources/rigid_references \
  --log-root logs/rigid_references --workers 8
```

The statistics CLI resumes per episode; the physical-reference CLI resumes
verified episode payloads and stores independent logs. Long preparation runs
should be launched with `nohup` and redirected logs.

Append Hydra overrides for hardware, batch size or experiments. To initialize
from task weights while resetting optimizer, scheduler and epoch state, pass
`resume_from=/path/to/model.ckpt resume_weights_only=true`. Normal checkpoint
resume retains training state. Keep the saved configuration with the weights.

## Inference

Use the checkpoint's complete `.hydra/config.yaml`, or export `cfg` from its
`hparams.yaml`. An experiment fragment or default YAML may not match the model.
The loader restores weights strictly and skips backbone initialization downloads.

```bash
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video input.mp4 --full-video --output predictions.npz
```

`--full-video` samples presentation timestamps and merges overlapping windows.
For a CFR clip, use `--frames 32 --start-seconds 0` instead; sampling follows
the config's `target_fps`. RGB frames are letterboxed to `data.image_size`.
`--device cpu` selects CPU inference. Tensor input uses `--input` with RGB
uint8 `[B,T,3,S,S]` or `[T,3,S,S]`; caller-normalized floats require `--normalized`.
T must fit positional-encoding capacity (currently 128).

Outputs are NPZ, same-stem JSON and a `.done` success marker. Existing outputs
are not overwritten. Hands are ordered left, then right; predictions use the
image-aligned substep at the input token rate, usually 10 Hz.

| Field | Meaning |
|---|---|
| `contact_prob` | Contact probability `[B,T,2,778]`; contact threshold 0.5 |
| `proximity_prob`, `distance_m` | Proximity encoding and decoded distance in metres |
| `force_normalized_10hz` | Normalized force vectors `[B,T,2,778,3]` |
| `hand_visibility_prob_10hz` | Optional per-hand visibility probability `[B,T,2]` |
| `source_frame_indices`, `target_times_s` | Source-video alignment |

## Evaluation

```bash
tactohand-eval --config model.yaml --ckpt model.ckpt \
  --datasets dexycb --split-by-dataset dexycb=test \
  --input-source full --metrics contact proximity --output outputs/evaluation
```

Use `--limit-batches 1` for a smoke check, `--processed-root` to relocate data,
and `--list-metrics` to inspect units. Preserve the saved `data.datasets` and
readable normalization statistics for the full training mixture, even when
evaluating one dataset; changing that list changes force scale.

| Dataset | Protocol |
|---|---|
| DexYCB / HOCap | `dexycb=test` / `hocap_multiview=test`, full-frame input |
| ARCTIC P2 | `arctic=test`, `--views ego_view --subject-filter s05`, full-frame input |
| MOW | `mow=test`, `--input-source handcrop`; repeat the static image for temporal input |
