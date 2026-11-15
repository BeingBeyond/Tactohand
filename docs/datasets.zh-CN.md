# 数据集

[English](datasets.md) | [简体中文](datasets.zh-CN.md)

## 目录与读取

`PROCESSED_ROOT` 指向 prepared 数据集的父目录，每个数据集对应一个
`<dataset>_lerobot` 子目录，原始数据另行保存。`meta/info.json` 指定活动 manifest。
按声明的索引读取，不假设 Parquet 或视频的物理文件名。

- `core.data.processed_index.ProcessedIndex` 解析 episode、数据流和范围。
- `core.data.processed_compat` 提供共享读取视图。
- `core.tactile_labels` 读取标签，`training.data` 组装模型输入。
- 物体几何按字符串 `object_asset_id → object_geometry_id` 关联，类别标签不是唯一资产键。

用 `source_position` 和声明映射对齐视频、标签、几何与相机。
位姿行、源帧、packed 视频帧和 token 属于不同索引域，各数据流可有独立分片边界与偏移。

## 采样与监督

训练从有效原生帧选取起点，再按配置采样窗口，通常为 32 帧、10 Hz。
验证与评估固定采样种子。静态输入可以重复图像，几何与力有效性仍按声明时间轴读取。

计算损失和指标前应用手部 presence、接触/距离有效性及 force-valid 掩码。
仅接触来源不参与力和 rollout 损失。关节来源可保留力监督，并设置
`rollout_eligible=false`；混合或其他不满足条件的 batch 跳过 rollout。
在评估子集上解码力时，保留原训练混合的统计量。

## 几何与裁剪

世界几何单位为米，相机采用 OpenCV 坐标（x 右、y 下、z 前）。
`c2w` 将相机系映射到世界系，`w2c = inv(c2w)`。
Prepared MANO 参数已包含来源坐标变换，通过 `core/geometry/mano_layer.py`
按已安装的基底元数据解码。

裁剪内参统一用 `frame_intrinsics`，输入手侧统一用 `HandcropStreamIndex.input_side`。
框坐标遵守存储声明，旧 raw-mirror 画布与归一化观测坐标不能混用。
存在逐帧内参时须保留，校正后的 processed 像素也不能替换为原生鱼眼像素。
详见[标签格式](label_format.zh-CN.md)。

## 数据集约定

| 数据集 | 读取要求 |
|---|---|
| Arctic | Raw ego 可为畸变图像，processed ego 可已校正。标定按 subject/camera 关联，各流按自身区间读取。重采样只改变像素与 K，不改变世界几何或力。 |
| DexYCB | 源采集为 30 fps。遵守已安装左右手 presence，不沿用基线的 right-only 限制；使用安装后的 split 和实例资产键。 |
| H2O3D | OpenGL→OpenCV 和手侧平移调整属于数据准备，读取端不重复施加，也不翻转全部局部关节旋转。遵守 pose/trans 可用性。 |
| OakInk-v2 | 读取明确的 episode/view/frame 映射，旧偏移、视角轮转和六帧规则不能推广为全局读取规则。按来源核对图像平面。 |
| PROX | 真实采集为 5 fps，prepared 的 10 Hz 输入可重复帧。按 token/source 映射读取，不使用旧容器的 30 fps 声明。旧裁剪画布为 1920×1080，标签是 hand-scene。 |
| RICH | 采集为 30 fps，机械切块不代表独立序列。旧裁剪画布为 1080×790，使用 hand-scene 监督；检查几何前确认场景网格可用。 |

## 数据检查

完整处理前先运行有界检查：

```bash
python scripts/data_checks/validate_labels_v5.py --help
python scripts/data_checks/check_handcrop_consumer_contract.py --dataset dexycb
```

裁剪修改后，沿真实消费者路径验证 token/source 映射、内参、相机变换和视频帧选择。
生成器内部自洽无法发现双方共用的索引错误。求解发布与标签失效规则见[标签格式](label_format.zh-CN.md)。
