# 示例视频

[English](README.md) | [简体中文](README.zh-CN.md)

普通 `git clone` 即可获取 8 段 RGB 视频：ARCTIC、HOT3D、HOI4D、DexYCB 各两段不同序列，
每段最长 20 秒。素材仅包含输入视频，不含标签或预先计算的预测。

安装 TactoHand，下载 checkpoint 及配套 YAML 后即可运行：

```bash
tactohand-infer --config checkpoints/tactohand.yaml \
  --ckpt checkpoints/tactohand.ckpt --video examples/videos/arctic-box.mp4 \
  --full-video --output prediction_outputs/arctic-box/predictions.npz
tactohand-view
```

替换视频路径即可体验其他素材。[在线演示](https://huggingface.co/spaces/zhenqis123/TactoHand)
也提供相同的 8 段视频，可直接点击选择。

| 数据集 | 视频 | 时长 |
|---|---|---|
| ARCTIC | `arctic-box.mp4`、`arctic-scissors.mp4` | 各 20 秒 |
| HOT3D | `hot3d-racket.mp4`、`hot3d-bowl.mp4` | 各 20 秒 |
| HOI4D | `hoi4d-car.mp4`、`hoi4d-kettle.mp4` | 各 20 秒 |
| DexYCB | `dexycb-can.mp4`、`dexycb-drill.mp4` | 2.4 秒、2.47 秒 |

视频采用无音轨的 H.264 MP4，保留已准备 RGB 视图的画面和帧率。
截取范围遵循数据清单中的序列边界；`manifest.json` 记录原序列、视角、序列内起始帧、
时长及 SHA-256，`previews/` 保存各视频中间帧的缩略图。

来源：[ARCTIC](https://arctic.is.tue.mpg.de/)、[HOT3D](https://facebookresearch.github.io/hot3d/)、
[HOI4D](https://hoi4d.github.io/)、[DexYCB](https://dex-ycb.github.io/)。
视频片段遵循各来源数据集的使用条款，仓库代码的 MIT 许可不替代这些条款。
