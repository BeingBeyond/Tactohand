# Datasets

[English](datasets.md) | [简体中文](datasets.zh-CN.md)

## Download

[Download from Baidu Netdisk](https://pan.baidu.com/s/1K3em0o7q0EykbVuGLRGRBQ?pwd=th26) (access code: `th26`; no expiry).
The dedicated release contains 24 prepared datasets with tactile labels and the
matching training resources. Each dataset's `meta/info.json` selects its active
version. Point `PROCESSED_ROOT` at the downloaded `data/` directory.
Download each selected `<dataset>_lerobot/` directory in full, preserving its
relative paths, including `meta/`, tables, object resources and video streams.
The published data already contains labels and hand crops; no raw-data conversion
or label regeneration is needed for training.

```text
dataset/
  README.md
  data/<dataset>_lerobot/
  training_resources/tactohand_v1.tar.gz
```

## Training resources

Download `training_resources/tactohand_v1.tar.gz` from the same folder and extract
it into `training_resources/`. The extracted `tactohand_v1/` contains contact and
force loss priors, fresh-training and model-output normalization, and multi-body
physical references (about 2.10 GB compressed). Sequence splits and the object
allowlist ship with the code. MANO model files and DINOv3 ViT-L/16 backbone
weights must be obtained separately; they are not included in this download.
See the [installation](usage.md#setup) and [training setup](usage.md#training)
for dependencies, extraction, environment variables and validation commands.

## Layout and readers

Set `PROCESSED_ROOT` to a parent containing one `<dataset>_lerobot` directory
per prepared dataset. Raw datasets remain external. Each dataset's
`meta/info.json` identifies its active manifest; use declared indexes rather
than assuming physical Parquet/video filenames.

- `core.data.processed_index.ProcessedIndex` resolves episodes, streams and ranges.
- `core.data.processed_compat` provides shared reader views over prepared storage.
- `core.tactile_labels` reads labels; `training.data` assembles model inputs.
- Object geometry uses the string join `object_asset_id → object_geometry_id`;
  class labels are not unique asset keys.

Use `source_position` and declared mappings to align video, labels, geometry and
cameras. Pose rows, source frames, packed-video frames and tokens are different
index domains. Streams may have independent shard boundaries and offsets.

## Sampling and supervision

Training chooses eligible native-frame origins, then samples the configured
window (typically 32 frames at 10 Hz). Validation/evaluation use fixed sampling
seeds. Static input can repeat images, but geometry and force validity still
follow their declared timeline.

Apply hand presence, contact/proximity validity and force-valid masks before
losses or metrics. Contact-only sources contribute no force or rollout loss.
Articulated sources can retain force supervision with `rollout_eligible=false`.
Mixed or otherwise ineligible batches skip rollout. Preserve the original
training-mixture statistics when decoding force on an evaluation subset.

## Geometry and crops

World geometry uses metres; camera axes follow OpenCV (x right, y down,
z forward). `c2w` maps camera to world, and `w2c = inv(c2w)`.
Prepared MANO parameters already encode their source transforms. Decode through
`core/geometry/mano_layer.py` using installed basis metadata.

Read crop intrinsics through `frame_intrinsics` and hand selection through
`HandcropStreamIndex.input_side`. Crop coordinates follow declared storage:
legacy raw-mirror canvases and normalized observation coordinates are distinct.
Retain per-frame intrinsics when present. Rectified processed pixels and native
fisheye pixels are not interchangeable; see the [label format](label_format.md).

## Dataset-specific rules

| Dataset | Required handling |
|---|---|
| Arctic | Raw ego may be distorted while processed ego is rectified. Join calibration by subject/camera and resolve each stream's interval. Resampling changes pixels/K, not world geometry or force. |
| DexYCB | Source capture is 30 fps. Respect installed left/right presence; do not impose a baseline's right-only convention. Use installed splits and instance asset keys. |
| H2O3D | OpenGL→OpenCV and hand-dependent translation adjustments belong to preparation. Do not repeat them in readers or flip all local joint rotations. Respect pose/trans availability. |
| OakInk-v2 | Read explicit episode/view/frame mappings. Historical offsets, view rotations and six-frame rules are not global reader rules. Check image planes against source records. |
| PROX | Physical capture is 5 fps; prepared 10 Hz input may repeat frames. Use token/source mappings, not a container's old 30 fps declaration. Legacy crop canvas is 1920×1080; interaction labels are hand-scene. |
| RICH | Capture is 30 fps. Mechanical chunks do not imply independent sequences. Legacy crop canvas is 1080×790; use hand-scene supervision and verify scene meshes before geometry checks. |

## Data checks

After downloading and extracting the resources, run the release resource check
from the repository root with the environment variables in the
[training setup](usage.md#training):

```bash
python scripts/data_checks/check_training_resources.py \
  --processed-root "$PROCESSED_ROOT" \
  --resources-root "$TACTOHAND_TRAINING_RESOURCES"
```

The check verifies statistics/manifest bindings and physical-reference payloads.
If you change crops, also verify the actual consumer path: token/source mapping,
intrinsics, camera transforms and video frame selection. Producer self-consistency
alone cannot detect shared indexing mistakes. Solver publication and stale-label
rules are specified in the [label format](label_format.md).

Training statistics and multi-body physical references are distributed under
`training_resources/tactohand_v1` alongside `data/`; see the [training setup](usage.md#training).
