# 共享代码与模块职责

2026-11-04 职责重组后的目录约定。

| 实现层 | 负责 |
|---|---|
| `core/` | 共享存储契约、数据访问、几何、物理数值算法和通用呈现 |
| `training/` | 模型、学习数据管线、训练、推理、评估及 batch/预测展示 |
| `label_generation/` | 离线标签生产：求解适配、调度、阶段流程、审计、发布与安装 |

| 操作目录 | 负责 |
|---|---|
| `scripts/` | 对数据、模型和实验执行操作；具体配方放各业务 `recipes/` |
| `tools/` | 开发回归、依赖检查、打包和工作区维护 |

按目的划分 scripts/tools，文件语言、运行时间或是否使用 GPU 不作为依据。
正式模型指标评估是业务操作；用合成输入断言指标正确属于开发回归。
顶层启动配方已按 training/evaluation/labels/profiling 分入对应业务目录。

```text
scripts/
  training/run_experiment.sh
  training/recipes/{tactohand,tactohand_wrench_32f,hope_baseline}/
  evaluation/recipes/{tactohand_wrench_32f,interfield_table1}/
  labels/{solve,publish}.py
  labels/recipes/
  profiling/recipes/tactohand_wrench_32f/
  data_checks/
  visualization/
tools/
  checks/
  packaging/
  workspace/
  run_checks.sh
```

## core 边界

- `io/` 提供文件、视频和底层索引能力；processed 地址与序列表解释归 `data/`。
- `data/` 读取 manifest、episode、相机、资产、帧映射，不选择学习窗口或监督资格。
- `geometry/` 包含 MANO/mesh、相机值类型、投影、力坐标系和质心/惯量计算。
- `physics/` 接收显式状态、参数、几何和时间轴；不读取数据集或选择数据集策略。
  通用刚体/关节参数类型在 `parameters.py`，多体计算在 `multibody.py`。
- `tactile_labels/` 拥有标签存储、物化、handcrop 与 token 索引契约。
- `visualization/prepared.py` 提供 `WindowView`、`ReviewContext` 和场景渲染；
  context 接收显式根路径。相机类型在 geometry，相机读取在 data，视频编码在 io。
- 数据审计 CLI 在 `scripts/data_checks/`；共享读取/校验在 `core/data/inspection.py`。

## training 边界

共享学习数据在 `training/data/`：Dataset、DataModule、采样、padding 和增强。
学习监督/rollout 资格由这里与任务定义，不通过底层 reader 丢弃可用标签。
共享 schedule 在 `training/runtime/`，通用生命周期回调和 loss 分别在 callbacks/losses。

模型包只拥有网络、Lightning 任务、特有目标和运行适配。同级模型不互相导入。
`model.py` 及参数属性名保持稳定；通用 force frame 和 wrench 聚合已移至 core。

`training/evaluation/` 负责标准视图、指标、报告及外部预测器。
`tasks.py` 是惰性任务装配边界，模型加载及专属视图构造在任务 runtime adapter；
当前 native checkpoint 评估仍支持 TactoHand。官方 ARCTIC InterField SF/LSTM、
EgoPHI、HACO、ContactPrompt 的输入、覆盖率与外部资源协议保持。
`training/visualization/` 将 batch/预测适配为共享场景，业务库不导入可视化脚本。

## label_generation 边界

该独立应用包负责调度、求解阶段和发布。`adapters/` 构造数据集求解输入，
选择参考质量、完成缺失输入及平滑策略；表/mesh 读取调用 core/data，
惯量与动力学调用 core/geometry、core/physics。重试/续跑仍使用既有检查点协议。
`scripts/labels/solve.py` 和 `publish.py` 是薄 CLI，recipes 提供平台参数。
离线生成代码进入选定源码包，保持在默认推理 wheel 之外。

## 依赖方向与兼容

```text
scripts/<业务>/recipes → 通用业务 CLI
scripts → training / label_generation / core
tools → 被检查的实现
training → core
label_generation → core
```

core 不导入其他实现层或操作目录；training 与 label_generation 通过标签文件
交换数据，不互相 import。数值 physics 不访问数据集，读取层不依赖呈现层。
共享训练代码只在显式惰性装配边界选择任务实现。
`tools/checks/check_architecture.py` 检查函数内 import、内部模块路径和 Hydra target。

三个 console 命令及 model.py 路径保留。已保存的旧 DataModule `_target_`：
`training.tactohand_wrench.datamodule.SubstepForceDataModule` 和
`training.tactohand_wrench.data.datamodule.SubstepForceDataModule` 均薄转发到
`training.data.datamodule`。新配置使用共享路径。旧 visualization camera 导入
只重导出 data/geometry 唯一实现。

配方移动保持参数、环境变量和真实退出码；不改变模型默认值、loss、label schema、
数值精度、积分步长或 checkpoint 键。源码/wheel/sdist 范围由
`packaging/public_files.toml` 控制，仓库外安装由 distribution 回归检查验证。
全仓 pyright 仍有存量诊断；运行回归通过不表示全仓类型通过。

## 几何与像素

- 世界几何单位为米；`c2w` 表示 camera→world，`w2c = inv(c2w)`。
- 相机轴采用 OpenCV 约定：x 右、y 下、z 前。
- MANO 世界轨迹通过 `core/tactile_labels/mano_tracks.py` 获取。
- 投影和 overlay 复用 `core/visualization/`，应用脚本不重复实现相机数学。
- 内参与视图关系、视频帧和源帧映射由 reader 解析，不能假设各流同名文件共享区间。
- handcrop 的 box 使用 raw-mirror 坐标，K 统一走 `frame_intrinsics`；逐帧 K 不能简化为固定 K。
- 物体几何解析复用 `instance_geometry_ids`，按字符串资产键关联；类别 label 不是唯一键。

## 图像与输出

- OpenCV 绘制和面板拼接使用 BGR、uint8；reader 的 RGB 在边界明确转换。
- 视频输出使用共享 H.264/FFmpeg 编码工具。
- 接触、距离和力色图及图例复用共享实现。
- 检查几何时走真实训练/评估的消费者路径，避免仅验证生成器内部自洽。

## MANO 资产与 basis

模型实体位于 `assets/mano/stock/models/`；不同的 corrected 左手模型位于
`assets/mano/corrected/models/`。第三方链接已更新到 stock；stock 内的平铺文件名
仅为第三方兼容链接。共享加载边界是
`core/geometry/mano_layer.py`。读取端不自行修改 shapedirs 或已准备的参数。
图文解释见 [MANO basis](mano_basis.html)。

模块索引见 [core](../core/README.md) 和 [training](../training/README.md)。
`core/` 不导入 `training/`、`scripts/`、`tools/`；模型参数名和训练入口保留。
