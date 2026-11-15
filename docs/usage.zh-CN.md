# 使用指南

[English](usage.md) | [简体中文](usage.zh-CN.md)

<a id="setup"></a>

## 安装

使用 Python 3.11、PyTorch 2.10、torchvision 0.25，并选择匹配的 CUDA 构建。
按工作流安装依赖：

```bash
python -m pip install '.[inference]'
python -m pip install -e '.[train,video]'
```

视频推理需要 FFmpeg、ffprobe。`video` 依赖组安装 TorchCodec 0.10，须与
PyTorch、CUDA、FFmpeg 匹配；`data.video_backend=cv2` 可选择 OpenCV。
完整标签求解需单独准备 Newton/Warp 环境，`labels` 仅提供数据和网格基础库。

| 资源 | 配置 |
|---|---|
| Prepared 数据集 | `PROCESSED_ROOT` 或 `data.processed_root`，指向 `<dataset>_lerobot` 的父目录 |
| 骨干初始化 | `PRETRAINED_PATH` 或 `model.wrench.pretrained_path`，指向匹配骨干的本地权重 |
| MANO 模型 | `MANO_ASSETS_ROOT`，目录内含 `models/MANO_LEFT.pkl` 和 `models/MANO_RIGHT.pkl` |

几何相关工作流还需从原仓库安装
[manotorch](https://github.com/lixiny/manotorch)：

```bash
git clone https://github.com/lixiny/manotorch.git
python -m pip install -e ./manotorch
```

本项目使用的原始 MANO 文件存在已知的
[左手形状基底问题](https://github.com/vchoutas/smplx/issues/48)：
左手形状基底 `shapedirs` 的 x 分量未做镜像，导致相同形状参数可能生成不一致的双手手型。
我们将该分量取反，使双手的形状变化保持一致。

`assets/mano/stock` 保存原始文件，`assets/mano/corrected` 保存修正后的文件。
加载器根据文件指纹判断是否需要修正，避免重复处理，无需手动修改 PKL。
已有拟合参数仍须使用与其匹配的基底；更换模型文件不会自动转换这些参数。

## 预测看板

在浏览器中上传视频，即可查看预测的接触、距离和力，以及每只手上方的可见性。

可直接使用[在线 Demo](https://huggingface.co/spaces/zhenqis123/TactoHand)，上传不超过 30 秒、100 MB 的视频，
左侧播放视频，右侧通过手部网格查看预测，支持同步播放、旋转缩放和逐帧可见性，
也可下载完整结果。免费 GPU 请求可能需要排队；本地运行方式见下文。

### 启动看板

从以下任一位置下载 `tactohand.ckpt` 及配套的 `tactohand.yaml`：

- [Hugging Face](https://huggingface.co/zhenqis123/TactoHand/tree/main)
- [百度网盘](https://pan.baidu.com/s/1iG-hTmECh_FrkJKIgWdSiQ?pwd=898f)（提取码：`898f`，永久有效）

将两个文件放入 `checkpoints/` 目录，然后启动看板：

```bash
mkdir -p checkpoints
cp /path/to/downloads/tactohand.ckpt checkpoints/
cp /path/to/downloads/tactohand.yaml checkpoints/
tactohand-view --port 8765
```

打开 [http://localhost:8765](http://localhost:8765)，在 **Server** 面板选择 checkpoint
和设备，点击 **Apply settings**。然后在 **New prediction** 区域上传视频并点击
**Predict**。可播放视频或拖动时间轴查看结果，在 **Prediction run** 中切换已保存的结果。

**Client** 只配置服务器地址；**Server** 管理 checkpoint、设备和 **Accept new predictions**
开关。关闭该开关后暂停接收新请求，运行中及已排队的任务仍按原配置继续执行。

MANO 模型放在其他位置时，可指定目录：

```bash
tactohand-view --port 8765 --mano-root /path/to/mano
```

<a id="remote-inference"></a>

### 使用远端 GPU

在 GPU 机器上按上述方式放好 checkpoint 和 YAML，再从仓库目录启动服务：

```bash
mkdir -p logs
nohup python -u -m training.visualization.inference_viewer prediction_outputs \
  --api-only --host 127.0.0.1 --port 8766 \
  --checkpoints-root checkpoints --mano-root /path/to/mano --device cuda:0 \
  > logs/inference_api.log 2>&1 < /dev/null &
```

在本机将 `USER@GPU_HOST` 替换为 SSH 登录地址，建立隧道并保持此终端打开：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18766:127.0.0.1:8766 USER@GPU_HOST
```

另开一个本地终端，从仓库目录启动看板：

```bash
python3 -m training.visualization.inference_dashboard --port 8765
```

打开 [http://localhost:8765](http://localhost:8765)，在 **Client** 面板输入
`http://127.0.0.1:18766` 并点击 **Connect**。在 **Server** 面板管理 GPU 服务器的
checkpoint 和设备，再到 **New prediction** 区域上传视频。使用期间保持 SSH 隧道连接。

## 训练

准备[数据集](datasets.zh-CN.md)、骨干权重和 MANO，先查看组合配置再启动。
本分支默认 `experiment=tactohand data=tactohand`，输入为 256px。

```bash
export PROCESSED_ROOT=/path/to/prepared
export PRETRAINED_PATH=/path/to/backbone/model.safetensors
export MANO_ASSETS_ROOT=/path/to/mano
tactohand-train --cfg job --resolve
mkdir -p logs
nohup tactohand-train > logs/train.log 2>&1 &
```

在命令后追加 Hydra 参数可调整硬件、batch 或实验。从任务权重初始化并重置优化器、
调度器和 epoch 时，传入 `resume_from=/path/to/model.ckpt resume_weights_only=true`。
正常 checkpoint 续跑保留训练状态。配置须与权重一同保存。

## 推理

使用 checkpoint 对应的完整 `.hydra/config.yaml`，或从 `hparams.yaml` 导出 `cfg`。
实验片段和默认 YAML 可能与权重不匹配。加载器严格恢复参数，不下载骨干初始化权重。

```bash
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video input.mp4 --full-video --output predictions.npz
```

`--full-video` 按实际时间戳采样并合并重叠窗口；恒定帧率片段可改用
`--frames 32 --start-seconds 0`，按配置 `target_fps` 采样。
RGB 等比例缩放并补边到 `data.image_size`。`--device cpu` 选择 CPU。
张量输入使用 `--input`，支持 RGB uint8 `[B,T,3,S,S]` 或 `[T,3,S,S]`；
调用方已归一化的浮点输入须加 `--normalized`。T 不超过位置编码容量，目前为 128。

输出包括 NPZ、同名前缀 JSON 和 `.done` 成功标记，不覆盖已有文件。
手轴顺序为左、右，输出与图像对齐的子步，频率等于输入 token 频率，通常为 10 Hz。

| 字段 | 含义 |
|---|---|
| `contact_prob` | 接触概率 `[B,T,2,778]`，阈值 0.5 |
| `proximity_prob`、`distance_m` | 距离编码及米制距离 |
| `force_normalized_10hz` | 归一化力向量 `[B,T,2,778,3]` |
| `hand_visibility_prob_10hz` | 可选逐手可见概率 `[B,T,2]` |
| `source_frame_indices`、`target_times_s` | 源视频对齐信息 |

## 评估

```bash
tactohand-eval --config model.yaml --ckpt model.ckpt \
  --datasets dexycb --split-by-dataset dexycb=test \
  --input-source full --metrics contact proximity --output outputs/evaluation
```

`--limit-batches 1` 用于冒烟检查，`--processed-root` 可调整数据位置，
`--list-metrics` 查看指标与单位。即使只评估一个数据集，也应保留保存配置中的
`data.datasets`，并确保完整训练混合的归一化统计可读；改变该列表会改变力尺度。

| 数据集 | 协议 |
|---|---|
| DexYCB / HOCap | `dexycb=test` / `hocap_multiview=test`，全帧输入 |
| ARCTIC P2 | `arctic=test`，`--views ego_view --subject-filter s05`，全帧输入 |
| MOW | `mow=test`，`--input-source handcrop`；时序输入重复静态图像 |
