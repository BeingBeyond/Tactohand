<div align="center">

# TactoHand

[English](README.md) | [简体中文](README.zh-CN.md)

### Learning Geometry-Based Dense Hand-Surface Interaction Fields from Video

Ziheng Xi · Weibo Li · Jiayi Yu · Sipeng Zheng · Wanpeng Zhang<br>
Haiweng Xu · Hao Luo · Chuan Mao · Zongqing Lu

[![arXiv](https://img.shields.io/badge/arXiv-2610.39585-b31b1b.svg)](https://arxiv.org/abs/2610.39585)
[![Code](https://img.shields.io/badge/Code-Available-brightgreen.svg)](https://github.com/BeingBeyond/Tactohand)
[![Project Website](https://img.shields.io/badge/Project-Website-0072B2.svg)](https://zhenqis123.github.io/tactohand-site/)
[![Hugging Face Demo](https://img.shields.io/badge/Hugging%20Face-Demo-yellow.svg)](https://huggingface.co/spaces/zhenqis123/TactoHand)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Available-brightgreen.svg)](docs/usage.md#prediction-viewer)
[![Data](https://img.shields.io/badge/Data-Coming%20soon-orange.svg)](#release-plan)

[Release Plan](#release-plan) · [Citation](#citation)

</div>

TactoHand predicts contact, proximity and 3D force at each of the 778 MANO
vertices per hand from monocular RGB video. It combines frame features with
temporal attention and learns from geometry, motion and differentiable physics.

## Quick start

Try the [online demo on Hugging Face](https://huggingface.co/spaces/zhenqis123/TactoHand):
upload a video to run inference and explore synchronized hand-mesh predictions,
with no local installation. To run inference locally, follow the steps below.

Use Python 3.11 with matching PyTorch 2.10 and torchvision 0.25 builds.
Install FFmpeg and ffprobe for video inference, then install from the checkout:

```bash
python -m pip install '.[inference]'
```

Download the checkpoint and matching YAML from the
[prediction viewer guide](docs/usage.md#prediction-viewer). For datasets and MANO
assets, see [setup](docs/usage.md#setup).

```bash
mkdir -p prediction_outputs/demo
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video examples/videos/arctic-box.mp4 --full-video \
  --output prediction_outputs/demo/predictions.npz
tactohand-view
```

The repository includes [eight example videos](examples/README.md) from ARCTIC,
HOT3D, HOI4D and DexYCB, ready to use after cloning.

Open the printed address to inspect predictions or upload another video. The
viewer shows contact, distance, force magnitude and predicted hand visibility.
It can also run locally while a [remote GPU service](docs/usage.md#remote-inference)
performs inference. RGB-only inference does not require datasets or MANO geometry.

## Documentation

| Guide | Contents |
|---|---|
| [Usage](docs/usage.md) | Setup, training, inference, evaluation and visualization |
| [Datasets](docs/datasets.md) | Prepared data, sampling and dataset-specific conventions |
| [Label format](docs/label_format.md) | Tables, units, indexing and geometry contracts |
| [Development](docs/development.md) | Architecture, checks, packaging and contributions |

## Release Plan

- [x] Release the project website.
- [x] Release the source code.
- [x] Release pretrained checkpoints ([download](docs/usage.md#prediction-viewer)).
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
