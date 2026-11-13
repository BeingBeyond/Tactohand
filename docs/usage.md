# 使用指南

安装、推理评估与分发说明。

- [安装与资源](#installation)
- [模型与数据资源](#model-resources)
- [推理与评估](#prediction)
- [可视化入口与现有限制](#visualization)
- [分发与构建](#distribution)


<a id="installation"></a>

## 安装与资源
当前提供包配置和 wheel 安装入口，服务当前 TactoHand 网络、训练和评测代码。
尚未完成全新依赖环境的真实数据训练验收；本页不将所有旧任务或 baseline 视为已支持。
资源一致性检查使用直接内容比较。

### 安装分组

先建立 Python 3.11 虚拟环境，并按目标机器安装匹配的 PyTorch / torchvision。
包当前约束 Torch 2.10 与 torchvision 0.25 系列；CUDA wheel 来源由部署环境决定。

```bash
## 网络 Python API（不含 Lightning、数据读取或标签求解）
python -m pip install -e .

## 独立推理 CLI（配置、tensor/视频输入，不依赖训练框架）
python -m pip install -e '.[inference]'

## 当前训练与评测入口、数据读取、日志
python -m pip install -e '.[train]'

## 当前默认 TorchCodec 视频后端，额外依赖系统 FFmpeg
python -m pip install -e '.[train,video]'

## 可选可视化、标签数据/网格处理基础库
python -m pip install -e '.[train,visualization,labels]'
```

`video` 额外组对应当前验证环境中的 TorchCodec 0.10 系列，需与 Torch/CUDA/FFmpeg
匹配；未安装时可显式选择 `data.video_backend=cv2`，其结果/性能未在本轮重新对账。
`labels` 仅提供数据与网格处理基础库，不声称安装完整 Newton/Warp 求解环境。
MANO、Newton/Warp 和外部 baseline 的来源及完整安装仍待补充；
这些额外工作流不会因安装基础网络而被自动克隆。

当前几何读取仍使用 manotorch backend。实际读取 MANO 参数的数据前，
需另行安装审核来源的 manotorch，例如对已有上游源码执行
`python -m pip install -e /path/to/manotorch`（本地工作区路径为 `third_party/manotorch/`），并提供 MANO 模型文件。
该源码、受限模型资产及数据均不由此包分发。安装 `[train]` 本身不等于已满足
所有数据集的几何解码条件。

共享物理库位于 `core/physics/`，随 wheel 安装，统一通过 `core.physics` 导入。
标签求解默认将输出写入 `artifacts/label_solver/`。
本次目录整理保留模型/数据契约和 checkpoint 参数名。
默认 wheel 的文件范围与源码工作区不同：旧任务、baseline 和运维脚本留在研究源码，
见 [分发范围](usage.md#distribution)。`tactohand-train` 不加组覆盖时默认使用
当前 `tactohand` / `tactohand`，通用研究 Python 入口继续保留原行为。

### 外部资源与配置

| 用途 | 配置方式 | 行为 |
|---|---|---|
| labels-v5 数据混合根目录 | `PROCESSED_ROOT` 或 `data.processed_root` | 指向包含 `<dataset>_lerobot` 子目录的目录；默认不再引用本机挂载 |
| 单数据集原始 root 字段 | `DEXYCB_ROOT` 或 `data.root` | 仅单 root 模式使用；混合数据集按 `processed_root` 拼接子目录 |
| backbone 初始权重 | `PRETRAINED_PATH` 或 `model.wrench.pretrained_path` | 显式本地文件优先；缺文件给出配置提示 |
| MANO 模型 | `MANO_ASSETS_ROOT` | 指向含 `models/MANO_LEFT.pkl` 和 `models/MANO_RIGHT.pkl` 的目录；本地为 `assets/mano/stock` 或 `assets/mano/corrected` |
| MANO 旧数据源决策 | `MANO_BASIS_DECISIONS` | 仅无持久化决策的数据使用外部审核记录，不生成猜测记录 |
| 训练统计量与筛选名单 | `core/resources/tactohand/` 或显式配置覆盖 | 包内原样保存；无工作目录依赖；旧 run 的显式路径仍按原含义读取 |

`pretrained=true` 且未设置本地权重时，保留 timm 的预训练下载行为；离线环境请
提供匹配 backbone 的文件。`pretrained=false` 加 `pretrained_path=null` 用于明确的
随机初始化/小模型检查，不代表正式实验配置。

当前 TactoHand 完整 MANO mesh contact loss 使用包内统计量，不再要求无用的
HACO 336/84 回归器或 contact_data_all.npy。旧 reduced-mesh loss 仍要求与其表示
匹配的外部回归器，并保持原计算。旧任务配置没有批量改写。

只查看组合配置，不读取数据或下载权重：

```bash
tactohand-train --cfg job experiment=tactohand data=tactohand
tactohand-eval --help
tactohand-infer --help
```

选择性训练示例（实际执行前须准备数据/资产，按工作区规则后台运行）：

```bash
export PROCESSED_ROOT=/path/to/processed
export PRETRAINED_PATH=/path/to/matching_backbone/model.safetensors
export MANO_ASSETS_ROOT=/path/to/mano
tactohand-train experiment=tactohand data=tactohand \
  'data.datasets=[dexycb]' 'data.val_datasets=[dexycb]' \
  data.batch_size=1 data.num_workers=0 training.devices=1
```

历史 A800/4090 集群配方已归入 `scripts/training/recipes/`，仍保留其平台默认值；
此处的标准 Python/已安装命令入口不依赖这些默认值。
独立推理与 `--config` 评测用法、输出单位和边界见
[独立推理与评测说明](usage.md#prediction)。

### 当前环境与验证范围

2026-10-28 从现有 `v2c_env` 读取的环境版本如下。依赖范围不是多版本兼容性认证。

| 组件 | 观察版本 |
|---|---|
| Python | 3.11 |
| Torch / torchvision | 2.10.0+cu126 / 0.25.0+cu126 |
| timm / safetensors | 1.0.25 / 0.8.0 |
| numpy / scipy | 2.2.6 / 1.17.1 |
| Lightning / Hydra / OmegaConf | 2.6.1 / 1.3.2 / 2.3.0 |
| pyarrow / pandas | 23.0.1 / 3.0.1 |
| TorchCodec | 0.10.0+cu126 |
| TensorBoard | 2.20.0 |

本轮在现有依赖环境中构建 wheel 并安装到独立临时目录，从仓库外验证包文件、
配置读取和 CPU 网络路径；未新建一套完全隔离的依赖环境，未运行真实数据训练。
完整标签生成、外部 baseline、GPU 训练/评测和正式公开代码分发仍需后续验收。
2026-11-04 的真实 checkpoint CPU 推理及单 batch 评测补充验证见下方资源记录；
它不代表全量精度复现或全新环境验收。

已通过：

- wheel 包含共享物理库、配置 schema、Hydra YAML 与训练资源；不含本地数据/模型资产/归档树。
- 仓库外 `tactohand-train --cfg job` 和 `tactohand-eval --help`。
- 三个当前实验配置及环境变量覆盖；旧任务与可视化模块按需导入。
- tiny ViT、2 帧、32×32 输入的 CPU 前后向；force 输出 `[1,2,2,778,3]`，contact 输出 `[1,2,2,778]`。
- 缺数据目录/本地权重的可操作错误。
- mesh loss 无 HACO 资产时与旧权重一致；336/84 reduced-mesh loss 仍使用各自回归器且结果一致。

本轮构建使用 `python -m pip wheel --no-deps --no-build-isolation --wheel-dir <output> .`，
临时安装使用 `python -m pip install --no-deps --target <temporary> <wheel>`。
`--no-deps` 复用现有训练环境的依赖，不代表从零安装依赖也已验收。


<a id="model-resources"></a>

## 模型与数据资源

使用与 checkpoint 配对的完整配置。训练 run 的 `hparams.yaml` 将完整配置
放在 `cfg` 下，可将这一段导出为 `model.yaml`。已有完整
`.hydra/config.yaml` 的 run 可直接使用该文件。推理从 checkpoint 恢复
网络权重，评测可通过 `--processed-root` 指定数据位置。

准备数据的父目录下包含 `dexycb_lerobot`、`arctic_lerobot` 等子目录。
每个数据集的 `meta/info.json` 指向对应 manifest。物理力解码还需要
checkpoint 对应的力尺度及几何数据。

<a id="metric-mapping"></a>

### 论文指标的报告取值

以报告 `per_dataset[dataset][metric].value` 与 `.unit` 为准：

| 论文指标 | JSON metric | 单位处理 |
|---|---|---|
| Macro-F1 / Micro-F1 | `contact/f1_macro` / `contact/f1_micro` | 原始值为 0–1；百分数表乘 100 |
| ADE / ADE-100 | `proximity/ade_mm` / `proximity/ade_clamp100_mm` | 已是 mm |
| Distance-error acceleration | `proximity/accel_err_m_s2` | 已是 m/s² |
| Vertex / Active force MAE | `force/point_mae_n` / `force/active_mae_n` | N → 论文 mN，乘 1000 |
| Force angle | `force/angle_deg` | 已是度 |

**当前 `paper.tables` 便捷映射尚未与论文完全对齐**：proximity 的 `accel`
映射到 `proximity/accel_m_s2`（预测场平滑度），不是论文的距离误差加速度；
force 的 MAE 值仍为 N。不要直接将该对象复制进论文表格。
主 README 中的 split/view/input 参数描述论文协议；完整复现还需固定权重、
核对数据版本和覆盖范围，并完成全量评测。MOW 的论文 proximity/contact
分别使用 SF/T，不能用一个当前 T 模型替代这两列。

<a id="visualization"></a>

## 可视化入口与现有限制

| 目的 | 已有入口 | 当前情况 |
|---|---|---|
| 数据集 RGB/几何/标签查看 | `scripts/visualization/visualize_training_batches.py` | 研究源码有入口，但 manifest 数据上的共享 review context 未适配 |
| 数据集上的预测与 GT 对照 | `tactohand-eval --visualize N` | 需要数据集几何；共享相同的 review context 问题 |
| 用户视频推理与结果展示 | `tactohand-view` | 网页选择 checkpoint 和视频进行推理，并浏览已完成的 run |

在当前真实 DexYCB 数据上，构建 `core.visualization.prepared.ReviewContext`
会抛出 `FileNotFoundError: .../meta/observations.parquet`。共享 manifest reader
已可读取该数据；review context 仍直接读取旧物理路径。应先接入共享 reader，
再沿完整消费者路径验收视频、几何、label 行映射与画面投影。现有 handcrop
消费者检查通过不代表这个 gallery 入口可用。

另一个研究脚本 `scripts/visualization/visualize_pred_vs_gt.py` 读取 checkpoint
与数据集、依赖 CUDA，不能读取独立推理输出作为任意视频渲染器。
源码包当前没有选入上述独立可视化脚本；在完成适配与消费者验收后再纳入公开入口。

浏览器查看器优先使用 NPZ 中的 `source_frame_times_s` 或 `target_times_s`，
旧文件可用同名 JSON 的采样率和 `--start-seconds` 对齐。在仓库根目录执行
`tactohand-view` 后，打开终端打印的地址。可直接选择 checkpoint 和上传视频进行推理，
也可从列表浏览已有 run。默认扫描 `checkpoints/` 下同名的 checkpoint 和完整 YAML 配置，
优先选择 `tactohand.ckpt`，完成结果写入 `prediction_outputs/`。网页推理需要 FFmpeg 和 ffprobe，
使用重叠窗口处理整段视频，有 CUDA 时默认用 GPU。
视频路径从 JSON 的 `input` 读取，也可在页面中选择本地视频。
若浏览器不支持原始编码，可转码为 WebM，并在 JSON 中设置 `preview_video`。
视频相对路径以该 run 的目录为基准。服务优先使用 8765，端口被占用时会选择空闲端口。
VS Code Remote 环境可转发终端打印的端口。
默认从 `assets/mano/corrected` 读取模板，安装 manotorch 后可旋转、缩放 MANO rest pose。
可用 `--mano-root` 覆盖模板目录，或用 `--no-mano` 显示顶点网格。
手模与原视频并排显示，预测按 MANO 顶点顺序着色。
接触视图沿用项目网站的珊瑚色与蓝灰色。三角形内插值后的概率大于 0.5 时标为接触。
距离视图使用网站的对数色阶，可切换 25、50、100 或 250 mm 的显示范围。
力视图显示向量模长，有相机系物理力字段时以 N 显示，其余情况显示归一化幅值。
当预测包含 vertex-local 归一化力且提供了 MANO 模板时，可打开 **3D force vectors**。
箭头按 rest pose 的逐顶点局部基展开，长度与归一化力幅值成正比，随对应手模旋转。
两只手共用同一箭长尺度。可调节整体显示大小，也可指向或点按箭头查看顶点编号和幅值。

<a id="prediction"></a>

## 推理与评估
当前入口支持独立配置与 checkpoint，无需保留历史 run 目录。
安装条件与外部资源见 [安装说明](usage.md#installation)。

### 配置与权重

使用与 checkpoint 架构和数据语义匹配的完整 YAML。可以复制训练 run 的
`.hydra/config.yaml` 到任意位置；单个 `conf/experiment/*.yaml` 不是完整配置。
也可对匹配该 checkpoint 的实验选项导出组合配置：

```bash
tactohand-train --cfg job --resolve experiment=tactohand data=tactohand > model.yaml
```

导出默认配置不能替代核对 checkpoint 的实际架构、图像大小、temporal_fusion、
proximity_tau、力坐标系等。加载时关闭 backbone 初始权重加载与下载，直接严格加载
checkpoint 的预测参数。支持当前 Lightning `state_dict` 的 `network.*` 权重和
裸网络 state_dict；缺失/形状不匹配会失败，不用随机权重补齐。

### 推理

```bash
python -m pip install -e '.[inference]'
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --input frames.npz --output outputs/prediction.npz
```

`--input` 支持 `.npy` 或含 `images` 的 `.npz`：

- RGB `uint8`，形状 `[B,T,3,S,S]` 或 `[T,3,S,S]`；S 必须等于配置 `data.image_size`。
- float 输入必须显式加 `--normalized`，表示调用方已按训练方式归一化；不自动猜测 0–1 像素。
- T 不超过网络位置编码支持的长度（当前 128）；正常使用训练时的窗口长度和采样率。

可自行生成没有数据许可依赖的输入来验证接口（不用于精度判断）：

```python
import numpy as np
from omegaconf import OmegaConf
cfg = OmegaConf.load("model.yaml")
size = int(cfg.data.image_size)
length = int(cfg.data.get("sequence_length", 32))
images = np.random.default_rng(0).integers(
    0, 256, (1, length, 3, size, size), dtype=np.uint8)
np.savez_compressed("frames.npz", images=images)
```

也可以读取一个视频窗口：

```bash
tactohand-infer --config model.yaml --ckpt model.ckpt \
  --video clip.mp4 --frames 32 --start-seconds 0 \
  --output outputs/video_prediction.npz
```

视频按 CFR 解释，以配置的 `target_fps` 采样，选择最近的源帧；源帧索引随输出保存。
输出同时保存请求的采样时间 `target_times_s`，供 HTML 查看器同步播放。
使用 `--full-video` 时按实际视频时间戳采样，处理整段视频并合并重叠窗口。
入口不自动认证视频是否恒定帧率；VFR 数据请由调用方按时间戳生成 tensor 输入。
视频顺序解码以保持帧序号，不依赖随机 seek；从长视频的后部读取窗口可能较慢。
RGB 转换后复用训练的 `letterbox_square`；窗口越过结尾直接报错，不偷偷补帧。
应选择与模型输入习惯相符的全帧或 handcrop 视频，不自动推断裁剪框。

输出是 NPZ、同名 JSON 说明和 `.done` 成功标记。已有输出不会被覆盖。
JSON 记录配置/权重/输入位置、形状、单位、手顺序和预处理方式。

| 字段 | 形状 | 语义 |
|---|---|---|
| `contact_prob` | `[B,T,2,778]` | `sigmoid(gate) * sigmoid(conditional_logit)`，与部署/评测定义一致 |
| `contact_gate_prob` | `[B,T,2]` | 每手 engagement 概率 |
| `contact_cond_prob` | `[B,T,2,778]` | 条件 contact 概率 |
| `proximity_prob` | `[B,T,2,778]` | 网络 proximity 编码 |
| `distance_m` | `[B,T,2,778]` | `-ln(max(p,1e-6))/proximity_tau`，米，与评测解码一致 |
| `force_raw_10hz` | `[B,T,2,778,3]` | 原始归一化力头输出 |
| `force_normalized_10hz` | `[B,T,2,778,3]` | 按配置施加 engagement gate 后的归一化输出；尚未施加物理单位 clamp |
| `vertex_force_camera_n_10hz` | `[B,T,2,778,3]` | 仅提供下面所需几何/尺度后生成的相机坐标力，N |

手顺序为 left、right。默认输出当前图像对齐的 substep 0，采样率为输入 token 速率，
通常 10 Hz；不通过复制或 GT 拼接伪造完整 30 Hz 序列。
standalone 首发路径支持当前两手、contact/proximity、vertex-force、S=1 模型；
旧 S=3 checkpoint 使用评测入口，`raw_local_mse` 诊断配置不在此推理范围。
contact-only checkpoint 的 JSON 会标记 `force_head_supervised=false`，不能将其力输出
视为受力监督训练的预测。

### 可选物理力输出

单凭 RGB 与 checkpoint 不能确定局部力轴的相机方向；旧 checkpoint 也没有保存
datamodule 计算的归一化尺度。接口不会默认尺度为 1 或凭数据集名字推测它。
使用 `--geometry geometry.npz` 明确提供：

| 输入字段 | 条件与形状 |
|---|---|
| `point_force_scale` | 必填；该 checkpoint 训练实际使用的正有限尺度，标量或每 batch 样本一个值 |
| `vertex_local_yz_camera_10hz` | `vertex_force_frame=vertex_local` 时必填；`[B,T,2,778,2,3]`，相机坐标中的正交单位 y/z 轴 |
| `hand_present_10hz` | 可选；`[B,T,2]`，bool/0–1，提供时施加相同 presence mask |

尺度应来自对应训练数据平面的 `SubstepForceDataModule.point_force_scale`，
不是 force class-count 文件，也不是随便选择的对象质量。局部轴应来自输入同帧的
手部几何；接口复用 `force_local_to_camera` 和训练的力幅值 clamp，
顺序为 scale → gate → 坐标转换 → 可选 presence → clamp。
未提供 presence 时不会猜测手是否存在。没有几何输入时仅返回归一化力，
不会产生标为 N 的相机坐标字段，也不会进行 rollout。

### 独立评测

```bash
python -m pip install -e '.[train,video]'
tactohand-eval --config model.yaml --ckpt /path/to/model.ckpt \
  --processed-root /path/to/processed --datasets dexycb \
  --metrics contact proximity force --device cuda:0 \
  --output outputs/evaluation
```

仍使用现有 reader、split/mask、预测器和指标定义；`--config` 模式要求实际 checkpoint
文件，不能使用只有 run 目录才能解析的 `last`/monitor 前缀。
`--processed-root` 与 `--labels-root` 可以覆盖数据位置；不修改原 YAML。
评测不会加载 backbone 初始化权重，也不初始化只服务训练 loss 的统计量/回归器。
它仍需要真实标签、数据解码和几何消费路径所需的资源。

力的归一化尺度取自原训练混合的 `label_stats`。新 `--config` 模式要求原配置
`data.datasets` 中的全部归一化统计可读，即使只评测其中一个数据集；未评测数据集
不需要为此加载视频或建立训练窗口。缺统计会明确报错，不能通过删减训练数据集列表
来偷偷重算另一套尺度。旧 `--run` 的历史缺失跳过策略保留，并在报告中标明策略。

报告写入 `--output/<timestamp>/tactohand/`。未给 output 时，独立模式使用当前目录
`outputs/eval/`。报告包含 `config_path`、有效数据根目录和原有评测选项。
现有 `--run ... --ckpt last` 调用方式保留；`--list-metrics` 可单独使用。

### 验证范围

```bash
python tools/checks/check_prediction.py --output artifacts/release_checks/prediction.json
```

CPU 检查使用随机 tiny 模型与合成 RGB/局部轴/标签，验证 checkpoint 严格加载、
推理/训练消费者一致、视频颜色与 letterbox、新旧评测模式指标一致、无训练资产评测，
原训练归一化统计缺失的拒绝，以及禁止导入 trainer/Hydra/数据/物理模块时的独立推理路径。
不代表真实 checkpoint 的精度、真实数据集覆盖或 GPU 性能验收。

2026-10-28：7 项 CPU 契约检查通过；wheel 在独立临时目录安装后，从仓库外调用
`tactohand-infer` 成功，且禁止 trainer/Hydra/数据/物理模块导入时仍可运行；
已安装 `tactohand-eval --list-metrics` 可独立列出注册指标。
这些安装检查复用现有 Python 环境的依赖，没有宣称全新依赖环境验收。


<a id="distribution"></a>

## 分发与构建
默认 wheel 和源码包以 [public_files.toml](../packaging/public_files.toml) 为共同清单。
清单只控制打包，不删除研究工作区，也不决定 Git 仓库的公开可见性。

### 默认分发

| 内容 | 处理 |
|---|---|
| 当前训练/推理/评测入口 | 保留；`tactohand-train` 默认选择 `tactohand` + `tactohand` 数据配置 |
| `training/tactohand_wrench/` | 当前网络、模型损失与专属 runtime；不打入测试/诊断入口 |
| `training/data/`、`training/evaluation/`、`training/visualization/` | 学习数据、评测协议与模型结果展示 |
| 当前 Hydra 配置 | 分发白名单中列出的 TactoHand 实验和共享根配置/schema |
| 共享 `core` 库、labels-v5、可视化 | 保留；不复制另一套实现 |
| `core/physics/` | 共享物理实现随包安装，统一通过 `core.physics` 导入 |
| 小型统计量、allowlist 和归一化记录 | 保留 `core/resources/tactohand/` |
| 评测渲染依赖的 review helpers | 仅保留评估使用的共享渲染模块及 package markers |
| 通用 launcher、用户文档 | 源码保留 `scripts/training/run_experiment.sh` 与选定文档 |

只选择文件，不按“未被静态 import 命中”推断可以删除。共享根模块中仍保留部分
兼容 API；数据转换/发布写入等可选工作流并不因此成为已验证的默认安装功能。
`labels` 依赖组仍只是基础依赖，不自动提供完整求解工作流。

### 留在研究源码工作区

- 保留的 HOPE 基线训练任务与配置；旧 ContactModel 和 ONNX 链路已删除。
- HACO、EgoPHI、ContactPrompt、InterField 等外部 baseline adapters 和第三方仓库。
- 可选标签生成和性能测量入口；历史数据迁移、修复和专项分析脚本已清理。
- 集群平台 launcher、研究档案、论文/专利、agent 配置、数据、权重和生成物。

这些文件不从 main 删除。研究用户从完整研究源码工作区运行原入口，并自行准备
对应依赖/资产；Python 的 dependency extras 不会把它们自动补进当前模型 wheel。
外部 baseline factory 已拆至 `predictors_research.py`；当前预测器入口与计算保留。
在默认包里请求研究 predictor 会明确提示使用研究源码，而非静默用零预测替代。
旧 `HacoPredictor` 源码导入位置通过惰性兼容入口保留。

已安装的 `tactohand-train` 默认选择 256px 的 `tactohand` 实验与
`tactohand` 数据配置。显式 `experiment=` / `data=` 参数仍优先。

### 构建本地候选包

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir artifacts/build/wheels .
python setup.py sdist --dist-dir artifacts/build/sdist
python tools/packaging/build_public_source.py --output artifacts/build/source-preview.zip
```

三者使用同一清单。ZIP 只收集显式代码/资源/文档路径，不读取 `.git` 历史、
不扫描本地数据目录、不覆盖已有 ZIP、不访问网络或修改仓库可见性。
sdist 保留构建所需的清单与工具，解包后仍可构建同范围 wheel。
构建时清理的仅是 `artifacts/build/` 下旧的打包缓存，防止之前的宽范围 wheel 留下过期文件。

### 仓库外分发检查

```bash
python tools/checks/check_distribution.py \
  --output artifacts/release_checks/distribution.json \
  --log-dir logs/release_checks/distribution
```

检查从白名单源码 ZIP 解压出的目录构建 wheel 和 sdist，再从 sdist 重建 wheel，
逐文件核对库内容一致。临时安装后，在无工作区路径的进程中验证三个命令入口、
配置与资源读取、随机 tiny 网络 checkpoint 加载，以及 tensor 推理输出。
同时检查源码包中的文档链接与通用 launcher 配置导出，不启动训练。
报告逐步落盘，全部成功后写同名 `.done`；失败详情保留在独立日志目录。

该检查复用当前解释器的第三方依赖，不安装依赖、不联网获取权重，不等同于全新
依赖环境安装或真实 checkpoint/数据验收。完整发布前仍需分别完成这些检查。

这定义的是安装/源码候选包范围。将整个 Git 仓库公开仍会暴露其分支与历史，
不能用“包内未包含”替代仓库公开前的审阅。来源/许可、正式权重分发和最终发布仍待完成。
