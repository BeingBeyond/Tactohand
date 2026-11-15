# Example videos

[English](README.md) | [简体中文](README.zh-CN.md)

Eight RGB clips are included with a normal `git clone`: two sequences each from
ARCTIC, HOT3D, HOI4D and DexYCB. Every clip is at most 20 seconds long.
These are input videos, without labels or cached predictions.

After installing TactoHand and downloading the checkpoint and matching YAML:

```bash
tactohand-infer --config checkpoints/tactohand.yaml \
  --ckpt checkpoints/tactohand.ckpt --video examples/videos/arctic-box.mp4 \
  --full-video --output prediction_outputs/arctic-box/predictions.npz
tactohand-view
```

Replace the video path to try another clip. The same eight clips are selectable
in the [online demo](https://huggingface.co/spaces/zhenqis123/TactoHand).

| Dataset | Clips | Duration |
|---|---|---|
| ARCTIC | `arctic-box.mp4`, `arctic-scissors.mp4` | 20 s each |
| HOT3D | `hot3d-racket.mp4`, `hot3d-bowl.mp4` | 20 s each |
| HOI4D | `hoi4d-car.mp4`, `hoi4d-kettle.mp4` | 20 s each |
| DexYCB | `dexycb-can.mp4`, `dexycb-drill.mp4` | 2.4 s, 2.47 s |

Videos are H.264 MP4 with no audio, preserving the prepared RGB views and frame
rates. Clips follow episode boundaries from the published dataset manifest;
`manifest.json` records sequence, view, episode-relative frame offset, duration
and SHA-256. `previews/` contains a frame from the middle of each clip.

Source datasets: [ARCTIC](https://arctic.is.tue.mpg.de/),
[HOT3D](https://facebookresearch.github.io/hot3d/),
[HOI4D](https://hoi4d.github.io/) and [DexYCB](https://dex-ycb.github.io/).
These excerpts retain their source datasets' terms; the repository's MIT code
license does not replace those terms.
