<div align="center">

# TactoHand

[English](README.md) | [简体中文](README.zh-CN.md)

### 从视频学习基于几何的稠密手表面交互场

Ziheng Xi · Weibo Li · Jiayi Yu · Sipeng Zheng · Wanpeng Zhang<br>
Haiweng Xu · Hao Luo · Chuan Mao · Zongqing Lu

[![arXiv](https://img.shields.io/badge/arXiv-2610.39585-b31b1b.svg)](https://arxiv.org/abs/2610.39585)
[![Code](https://img.shields.io/badge/Code-Available-brightgreen.svg)](https://github.com/BeingBeyond/Tactohand)
[![Project Website](https://img.shields.io/badge/Project-Website-0072B2.svg)](https://zhenqis123.github.io/tactohand-site/)
[![Data & checkpoints](https://img.shields.io/badge/Data%20%26%20checkpoints-Release%20plan-orange.svg)](#release-plan)

[发布计划](#release-plan) · [引用](#citation)

</div>

TactoHand 从单目 RGB 视频预测双手各 778 个 MANO 顶点的接触、距离和三维力。
模型通过时序注意力融合图像特征，使用几何、运动和可微物理监督训练。

## 快速开始

使用 Python 3.11、匹配的 PyTorch 2.10 和 torchvision 0.25。
视频推理还需要 FFmpeg、ffprobe。在源码目录安装：

```bash
python -m pip install '.[inference]'
```

请从[预测看板指南](docs/usage.zh-CN.md#预测看板)下载权重及配套的 YAML 配置。
数据集和 MANO 资源的准备方式见[环境准备](docs/usage.zh-CN.md#setup)。

```bash
mkdir -p prediction_outputs/demo
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video input.mp4 --full-video \
  --output prediction_outputs/demo/predictions.npz
tactohand-view
```

打开终端打印的地址，即可查看预测或上传其他视频。看板显示接触、距离、力模长和
预测的双手可见性，也支持本地看板连接[远端 GPU 服务](docs/usage.zh-CN.md#remote-inference)。
仅 RGB 推理不需要数据集或 MANO 几何。

## 文档

| 文档 | 内容 |
|---|---|
| [使用指南](docs/usage.zh-CN.md) | 安装、训练、推理、评估与可视化 |
| [数据集](docs/datasets.zh-CN.md) | prepared 数据、采样及数据集读取约定 |
| [标签格式](docs/label_format.zh-CN.md) | 表结构、单位、索引与几何契约 |
| [开发指南](docs/development.zh-CN.md) | 架构、检查、打包与贡献方式 |

<a id="release-plan"></a>

## 发布计划

- [x] 发布项目网站。
- [x] 发布源代码。
- [ ] 发布预训练权重（[下载入口](docs/usage.zh-CN.md#预测看板)）。
- [ ] 发布数据集。

## 许可证

代码采用 [MIT 许可证](LICENSE)。

<a id="citation"></a>

## 引用

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
