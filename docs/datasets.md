# 数据集读取注意事项

这里只保留影响训练、几何和帧对齐的约束。数据已准备完成，不再维护下载、
转换、恢复及安装流程。各项来源于既有数据说明；旧分片统计和修复进度不能
视为当前安装状态。当前路径、帧映射和来源以安装元数据、`meta/provenance.json`
及 `core/data/processed_compat.py` 为准。

## 共用约定

- 使用共享 reader 获取帧、相机和几何，不根据文件编号推断 episode 对应关系。
- crop box 使用 raw-mirror 像素约定，crop K 通过 `frame_intrinsics` 读取；不要另写缩放公式。
- 位姿行、源帧和 token 是不同索引，按声明的映射关联；不能用“画面最像”代替映射。
- 已准备的 MANO 参数不再由读取端补历史坐标变换或偏置；basis 由 `core/geometry/mano_layer.py` 统一处理。
- split、手侧有效性和监督掩码以已安装表为准，不沿用旧转换脚本的默认值。

<a id="arctic"></a>

## Arctic

- raw ego 是畸变图像，processed ego 可以是校正图像；两者不能只按分辨率线性缩放对应。
- 相机标定按 subject 和 camera 关联，同一 view 编号不能跨 subject 假设固定物理视点。
- 历史 processed 视频存在 portrait/landscape profile。读取当前路径、尺寸和区间，不硬编码旧目录布局。
- full/raw/crop 的分片边界可能不同；即使文件名相同，也要按各自区间定位。
- 图像平面重采样会改变 K 和图像 bbox，不应改变世界 MANO、物体位姿或力标签。
- 几何检查走 `scripts/data_checks/check_handcrop_consumer_contract.py`；实际训练输入走 `scripts/visualization/visualize_training_batches.py` 查看。

<a id="dexycb"></a>

## DexYCB

- 源数据为 30 fps、多 RGB-D 相机、YCB 物体交互；序列可标注左手或右手，按已安装的手侧与 presence 读取，不能统一屏蔽左手。
- 官方 [toolkit](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py) 从每条序列的 `mano_sides[0]` 读取手侧；单个 baseline 的 right-only 约定不能传入共享训练 reader。
- 包含 MANO、关节、相机内参和物体位姿；dense contact/force 是后续生成监督。
- 训练使用安装后的 split 和时间轴，不依赖旧 `train_s0_29k.json` 等转换期文件。
- 资产几何按 `object_asset_id → object_geometry_id` 解析，不能用类别整数替代实例键。

<a id="h2o3d"></a>

## H2O3D

- 原始参数的 OpenGL→OpenCV 相机系转换曾涉及 `[x,y,z] → [x,-y,-z]`。
- MANO 坐标转换只改变 root/global 旋转，不能把 48 维 pose 的局部关节旋转全部翻转。
- 原始导入使用过不同的左右手平移偏置；该修正属于数据准备，读取已安装参数时不能重复施加。
- 单侧偏移先查手侧和源参数约定；双侧偏移先查帧映射、相机和投影链。
- 手部监督必须服从相应 pose/trans 的可用性标记。

<a id="oakink-v2"></a>

## OakInk-v2

- 历史问题涉及 raw 视频与 canonical/标签错位，也曾发现 canonical 像素损坏；不能默认任意一个视频平面永远正确。
- 修复曾针对具体相机、episode 和共享分片定义映射。旧偏移、跨视角轮转和六帧周期规则不能作为全数据集的 reader 补丁。
- 不要为了匹配错误 raw mirror 移动标签；也不要靠放宽像素误差阈值接受损坏帧。
- episode/view 的原始序列、相机和标注 ID 需要一致；共享分片中的 episode 长度以索引为准。
- 周期性像素重复或错位本身不足以推断真实采样率。
- 当前读取使用正式索引和训练消费者路径。旧下载状态、恢复 marker 和抽样比例不代表当前完整性。

<a id="prox"></a>

## PROX

- 真实采集为 **5 fps**，不能把旧容器的 30 fps 声明当作物理时间轴。
- 历史上的 18 个机械切块属于同一次连续采集；不要因旧片段编号重新划分序列。
- 已准备监督使用过 5 fps→10 Hz 帧堆叠，形如 `[0,0,1,1,…]`；读取 token/source 映射，不能用 30 fps 行算术替代。
- crop box 按 raw-mirror 1920×1080 编码，不是旧 processed 分辨率下的像素。
- 接触对象是 Vicon 场景，不能以旧 hand-hand 标签替换正式 hand-scene 标签。
- `solver_vertex_tracks(..., rows=...)` 的 rows 索引传入 `source_position` 的行轴，并非源帧号；帧堆叠时混用会产生半程错位。

<a id="rich"></a>

## RICH

- 官方采样率为 **30 fps**；不能仅凭相邻帧差异推翻标定时间轴。
- 历史 episode 是连续视频的机械切块，不应据此推断独立物理序列。
- crop box 采用 raw-mirror 1080×790 坐标，按实际尺寸通过共享 reader 获取 crop K。
- 使用 hand-scene tactile 监督；旧 hand-hand 版本已经退役。
- 旧安装缺少场景网格，不能据此声称已完成几何复算；如需复核，先确认当前安装是否提供对应场景资产。
