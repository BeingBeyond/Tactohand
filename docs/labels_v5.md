# TactoHand 标签存储格式（labels-v5）

正式数据位于 `_processed/data/<dataset>_lerobot`。入口是
`meta/info.json` 指向的 prepared manifest。manifest 的整数 ranges
将 episode、stream 和逻辑行映射到 Parquet 文件及物理偏移。
episode 可以跨文件和 row group。RGB、HMR、相机和 handcrop 保留各自的索引。

## Tactile family

| 组件 | 行单位 | 内容 |
| --- | --- | --- |
| `samples` | solver 行 | `contact`、`signed_distance` |
| `transitions` | solver 状态行 | 逐顶点世界系力、contact/environment wrench、目标物体的 COM 位姿与速度 |
| `timeline` | solver 行 | `solver_row`、`solver_time_s`、`source_position`、`tactile_flags`、`force_valid`、`target_object`、`closing_solver_row` |

- 紧凑 prepared manifest 在载荷组件中声明 `row_metadata: tactile.timeline`。
  三张表按 episode 和 solver 行对齐。读取器从 timeline 还原逻辑标签字段，
  兼容含重复字段的旧 prepared 版本。监督区间需要时从 timeline 生成。
  对旧标签编号等于源帧编号的稀疏静帧数据，组件可声明
  `solver_index_column: source_position`，以保留原标签编号与内容校验值。
- `timeline.tactile_flags` 的 bit 0/1 是左右手 presence，bit 2/3 是 contact 和
  proximity 有效位。`contact` 为 uint8，`signed_distance` 为 float16 米，
  截断到 ±0.25 m。每行按左右手各 778 个 MANO 顶点排列。
- `timeline.force_valid` 标记力监督是否有效。顶点力和 wrench
  使用 solver 的参考质量约定。参考质量、COM、惯量、`object_ids` 与
  target slot 对齐。Articulated 数据另外存储 generalized wrench 和 joint state。
- `source_position` 记录源视频帧坐标，包括求解器重采样时的分数位置。
  独立静帧允许标签稀疏覆盖原视频，按载荷行读取标签、按源帧位置读取 RGB 和相机。
  静帧的 solver 时间和结束行为空，每行可有独立的 object slot 与重力。
- 动态行的 `closing_solver_row` 只在同一目标参考系下有下一状态时存在。
  直接力监督可使用片段最后的有效行，rollout 还需要连续间隔和结束状态。
- episode 内嵌 tactile context 保存 topology、频率、物理参数与内容摘要。
  `tactile_sha1` 按 `solver_sample`、`source_position`、`flags`、
  `contact`、`signed_distance` 的顺序校验逻辑数组，独立于 Parquet 分片。
  `force_sha1` 覆盖该 episode 已声明的力、状态及附加物理字段。

## 训练和评测读取

训练从可解码的源帧选择片段起点，同一个 epoch 的样本数由各源数据配额控制。
从起点按模型的时间间隔取后续图像，RGB 映射、几何对齐和力监督分别核验。
当前模型每个视觉步对应最多三个 30 Hz solver 间隔。直接力预测只监督实际
与输入图像对齐的 solver 行，rollout 使用有效连续区间和同目标的结束状态。
稀疏视频沿用原视图的 causal hold，几何只在与实际显示帧对齐时有效。
验证和评测固定采样器种子，报告中记录 prepared manifest 和力尺度。

窗口缓存按 episode 有界保存标签，单目标全视图初始化可从行数和视频覆盖
建立窗口，多目标片段按明确的目标槽位分段。训练快速路径可只物化
单子步模型所需的 vertex-local 标签，完整相机系 GT 和 rollout 掩码仍由对应
solver 行提供。contact-only 配置同时屏蔽直接力和 rollout。

## 求解、发布和校验

`scripts/labels/solve.py` 输出 canonical solver NPZ。
`scripts/labels/publish.py` 由原始 solver 有效行构造目标槽位和监督区间。
多刚体目标优先采用标注对象，缺少对应标注时按该段实际受力物体的 wrench
能量选取。无效力行清零，逐顶点力仍为各物体贡献之和。
已转换数据中的历史参考系和力值按原发布结果保留。

`publish.py --install` 在 prepared 数据集中写入版本化 tactile components
与 episode context，保留原有 RGB、HMR、相机、对象和帧索引。先验证文件与
范围，再原子更新该数据集的 `meta/info.json`。旧引用保留用于回滚。
`scripts/labels/compact_tactile_release.py` 可对已有 prepared 分片做同样的投影，
逐行验证被移除的字段可由 timeline 还原，分片写完后生成候选 manifest 和验收记录。
重新求解改变数据 manifest 时，应更新依赖该版本的力统计和刚体参考。

`scripts/data_checks/validate_labels_v5.py` 通过实际消费者读取路径校验
native timeline 和 tactile/force 内容摘要。Handcrop 更新另需通过
`scripts/data_checks/check_handcrop_consumer_contract.py`。
`tools/checks/tactohand/test_wrench_rollout_identity.py` 用已发布力与状态验证
有效窗口的物理回放。

## 刚体参考

缺失的逐刚体参考量
在训练前离线恢复，写入独立 `rigid_reference_v1` 辅助目录。这是带版本和源指纹的辅助标签。每个 dataset/episode 包含 `bodies.parquet` 与原子
发布的 `manifest.json`。字段为 solver-row 对齐的逐刚体 COM 位置、旋转、
速度、external/residual wrench、逐帧世界重力、双手 778 顶点 owner（int16，
-1 表示无归属）和逐刚体有效位。object_ids 顺序与原标签索引一致。
environment 是从历史动力学参考和已发布力场恢复的残差外力，不是独立
测量的桌面支撑；不能用该残差作为真实接触力准确性的独立证据。
恢复需验证已发布目标状态、速度及合并 wrench。位姿插值区间及其 Gaussian
核邻域按原生成流程恢复逐刚体 `dynamics_valid`：无观测支撑的刚体为 neutral，
不产生 hand/environment wrench，不进入该刚体的 rollout；其它有效刚体继续
参与。持久化 `valid` 为原 force-valid 与逐刚体 dynamics-valid、恢复一致性
的交集。全部应恢复的刚体行及场景一致性检查通过后才发布 ready，不能用
neutral 标记掩盖真实的不一致。原 force 字段和有效标记均保持不变。
源标签哈希、数据 manifest、恢复版本和 payload
SHA256 用于失效检测；训练严格读取该目录，不运行归属查询或 force 求解。
准备入口为 `scripts/data_checks/recipes/prepare_rigid_references_80workers.sh`。
归属查询在 0.1 mm 并列时按 target 物体优先、否则最低 slot 确定性 tie-break。
退化网格的有限 BVH 结果默认直接信任（`trust_degenerate`），只有 BVH 查询
失败才做 float64 精确回退；需要旧的全量精确行为时传 `--exact-degenerate`。
每个 manifest 的 `report.ownership_policy` 记录本次使用的策略。这样避免在
物理不可分辨的边界或薄三角形上为逐三角形查询付出主要耗时。
该入口默认 80 workers，将 BVH 查询分配到可见 CUDA GPU；精确三角形回退
按物体跨帧批量处理。HOT3D/SHOW3D 辅助引用使用 observed-dynamics-v4 指纹，
其它完整轨迹数据集保留原指纹以复用已完成结果。

## MANO 几何迁移补充契约（2026-09-28）

规范的 MANO 解码入口为 `core/geometry/mano_layer.py`。迁移后的 HMR 用 corrected
PKL，布局为 `[trans3, full_axis_angle48, betas10]`，单位为米/弧度，
`flat_hand_mean=True`、`use_pca=False`、`center_idx=None`。消费端不得额外翻转
左手 shapedirs；入口根据已验证的模型指纹完成且仅完成一次必要的归一化。
原始数据适配器须显式声明原始基底，不能通过替换模型冒充参数转换。

尚未迁移的源参数可在 `meta/info.json.mano_source_basis` 持久记录解码选择：
`version=1`、`basis=stock|corrected`、`decision_status`、左右资产
`assets_sha256` 和 `decision_evidence`。用户指定的选择使用
`decision_status=user_confirmed`，证据须明确是用户选择而非独立来源认证。
读取顺序为已安装 `mano_contract` → 本地 `mano_source_basis` → 历史审计
registry；右手使用共用右手资产。该字段不表示参数已迁移或几何已通过认证，
视频目录等无关元数据变化不使此选择失效。

消费者检查须直接使用 labels 的 `source_position` 选择几何行和源帧，支持
单帧静态样本与 PROX 的重复帧时间轴；分段裁剪以 `frame_offset` 汇总覆盖。
`hmr.mano=false` 且声明 `mesh_vertices=true` 的根优先读取存储顶点，
`mesh_topology=mano_778` 时可复用 MANO faces，但不得解码占位 MANO 参数。

裁剪能力可声明 `file_frame_offset_source=from_timestamp`，使用
`round(from_timestamp * fps)` 定位 packed 视频；不得把已删除索引行留下的
间隙压成累计行数。规范化后每段独立文件的起始时间戳归零，保留该声明。
缺少物理 raw 视频时，已确认的 `crop_coordinate_space.raw_width/raw_height`
可作为裁剪坐标参考；这不声明 raw 视频已恢复或可读取。

`meta/info.json.mano_contract` 记录基底、姿态/平移约定、左右模型 SHA256、
来源证据、shape 策略和 `geometry_revision`。stock → corrected 的安装门槛是
所有 present 且有限的帧、每个顶点重解码误差的最大值默认 ≤1 mm。
用户于 2026-09-28 明确接受 HIC 当前最大约 1.8983 mm 的拟合结果，因此
**仅 HIC 使用 ≤2 mm**，其余数据集仍为 ≤1 mm；该例外写入来源决策的
`precision_acceptance`，拟合与安装均须核对。`mano_contract` 同时记录
`surface_error_limit_mm` 与实际 `surface_error_max_mm`。有无效或
超门槛帧的分片不得安装，全部已声明分片通过后才能安装该数据集。当前
拟合允许逐帧 beta 以满足表面精度，不宣称 subject beta 恒定。此 HMR
事务备份仅服务于几何迁移，不改变 §8 对派生标签重建的处理决定。

参数迁移或已确认的旧求解基底错误会置 `mano_dependent_labels.state=stale`；
训练读取端必须拒绝这些标签。新求解 run 的 `mano_geometry_contract.json`
绑定 `geometry_revision` 和每个 HMR 文件指纹。v5 publisher 对账后才能
清除 stale，同时继续执行既有 handcrop 发布检查。HIC 不生成物体受力，
使用 corrected MANO 重算手—手接触与有符号距离，保留原有非整数帧率
solver 时间和源帧的映射规则。未决来源和隔离候选不得标记为 corrected。

### Shared full-frame payloads and explicit crop-coordinate canvases (2026-10-20)

`raw_videos` may alias `videos` when the installed video was encoded at source
resolution, with identical image plane and frame order. Declare
`full_frame_video_source.kind=native_shared`, `same_resolution_as_videos=true`,
and `capabilities.raw_videos.shares_encoded_files_with=videos`; the independent
raw observation index still resolves every episode/view and packed offset.
A calibrated/resized processed source instead declares
`full_frame_video_source={storage: videos, kind: calibrated_processed,
native_resolution: false}` and does not imply that native raw is installed.

RGB storage dimensions and encoded `*_orig` box units are independent.
`beingh_human.crop_coordinate_space` is authoritative for stored geometry,
including when a physical/shared raw mirror exists. A global canvas retains
`raw_width`/`raw_height`; mixed canvases use `by_video_key`, mapping each exact
`observation.images.*` key to its own `raw_width`/`raw_height`. The encoding is
`observation_pixels_scaled_per_axis_to_declared_canvas`. Consumers must scale
through this declaration once; they must not substitute the shared video's
physical resolution for the coordinate canvas or treat rectified processed
pixels as original fisheye pixels. Video, MANO and stored crop intrinsics need
no rewrite when restoring this declaration.

A calibrated processed crop source may coexist with a separately installed
native raw index. Restoring native raw must not change the image plane used to
render or verify existing rectified crops. `capabilities.raw_videos` describes
actual native coverage independently of `full_frame_video_source`, which remains
the declared crop reconstruction source. Raw-only audits explicitly select the
independent raw index rather than the crop source.

For ReInterHand, native ego RGB retains its original 1024x1024 fisheye plane;
native Mugsy RGB is 667x1024. The processed 384x384 ego and 250x384 Mugsy streams
and their calibrated crop geometry remain in the processed pinhole plane.
`raw_video_encoding.calibration_path` points to a metadata attachment containing
per-episode/view native focal lengths, principal points, distortion model and
coefficients. Native intrinsics must not replace processed camera intrinsics.

### Handcrop geometry storage v2 (2026-10-20)

Geometry payloads use
`crop_data/hand/observation.images.<view_id>.<target_id>/chunk-XXX/file-XXX.parquet`.
`meta/cropped_video_observations.parquet` remains the authoritative segment
index: its path, row group, offset and row count are independent of video
chunk/file numbering. Each payload contains one view/hand stream. Writers pack
small segments with targets of 32,768 rows per row group and 262,144 rows per
file (a single larger segment may exceed these targets); segment boundaries,
video timestamps and observation frame indices are unchanged.

All datasets use the same Arrow schema: non-null `episode_index:int64`,
`view_id:string`, `target_id:string`, `frame_index:int64`, three float64 fixed
lists of length four (`crop_box_xyxy_orig`, `hand_bbox_xywh_orig`,
`hand_bbox_xywh`), and independent non-null `frame_valid:bool` and
`crop_valid:bool`. `cam_intr` is a nullable typed struct of float64
`fx, fy, cx, cy` describing the crop projection. A null struct preserves a
legacy absence of stored crop intrinsics; readers derive K through the shared
camera/crop path. It does not assert that K is zero. No opaque MessagePack
intrinsics are written in v2; shared readers continue to decode v1 MessagePack.

`beingh_human.crop_data.version=v2` declares the layout and field semantics;
Parquet schema metadata also records v2. `crop_coordinate_space.kind` is
`normalized_observation`: both `*_orig` fields use unbounded normalized
coordinates in the observation camera's projection plane (x and width divided
by image width, y and height by image height). The xyxy and xywh conventions
remain distinct. Negative/out-of-image windows are retained for black padding;
normalization must never clip them. `hand_bbox_xywh` remains in crop pixels.
This encoded plane is separate from the distortion and raster size of native
raw RGB. The shared crop canvas resolver returns a unit canvas for v2.

Migration must preserve masks, frame ordering, all non-geometry index columns,
intrinsic float64 values and the previous physical crop transform. It validates
all rewritten row groups and the actual shared consumer before/after install.
Old geometry trees, including unindexed historical payloads, are archived for
rollback. Existing MP4s, camera/HMR data and tactile/contact labels are not
rewritten by this storage migration.

### Handcrop geometry v3: derive crop intrinsics (2026-10-20)

v3 supersedes v2's duplicated `cam_intr` field. Canonical crop Parquet payloads
contain only the nine identity/frame, box and validity fields; no binary or
structured crop camera intrinsics are stored. Root layout, segment locators,
unbounded normalized `*_orig` boxes and crop-pixel hand bbox semantics remain
unchanged. Metadata declares `crop_data.version=v3` and
`intrinsics=derived_from_observation_camera_and_crop_box`.

The authoritative K is the observation camera's full-frame projection K,
resolved through `observation.camera_id` and `meta/cameras.parquet` in the
same image plane as `videos`. Denormalize the crop window using observation
width/height, then derive K with the shared `frame_intrinsics` /
`crop_adjusted_intrinsics` path and the indexed crop output dimensions. Native
raw fisheye K must not replace a rectified processed camera K. Padding outside
the image keeps the original crop-window origin.

Every valid stored legacy K is checked against this derivation before its
column is discarded (maximum component error 0.001 crop pixels). Differences
above that gate prevent publication. Legacy v1/v2 storage remains readable;
`HandcropStreamIndex.crop_intrinsics` returns no stored K for v3, so consumers
use the shared derivation. Migration preserves video, label, camera and HMR
payloads and archives the previous geometry for rollback.


### Processed dataset container v1 (2026-10-20)

The revision-5 label semantics are unchanged. A dataset whose `meta/info.json`
contains `format_version=processed_dataset_v1` stores only `meta/`, `data/`,
and `assets/`. The metadata root contains `info.json`, `provenance.json`,
`stats.parquet`, `tables/`, and `index/`. `info.json.manifest` points to the
integer index manifest; caches pin its revision. Supplemental model and
visualization outputs are preserved as registered assets without changing
their internal format. Full original trees remain outside `_processed` for
rollback; source symlinks are materialized into the published tree.

The three index families are episodes, frames, and per-modality data_ranges.
Each range has `episode, stream, start, count, path, offset`; bounds are
half-open and offset is a file-global row or video frame. Sparse camera frames
remain absent. Timestamps are auxiliary, and solver_sample, solver_transition,
and token retain their own domains plus source-frame mappings. Geometry frame
numbers follow canonical episode frames after translating packed video indices.
Crop v3 omits stored K; observation-camera K and normalized crop boxes define
crop intrinsics, independently of native raw distortion and resolution.

`core.data.processed_index` reads the normalized storage. `core.data.processed_compat`
provides legacy reader views derived from it, including row-group slices;
consumers must honor row_offset and num_rows. The container does not retain
standalone staging paths such as `meta/label_index.parquet` or `force_data`
paths. Labels and force streams are modalities under data/, with native
scientific contracts retained in provenance and the manifest. All-null optional
fixed vectors may use nullable variable-list encoding, retaining NULL values.
Predicted proximity scores retain their model-score semantics and do not become
revision-5 physical distances merely by sharing a container.

Native token contact_valid is distinct from contact-only training admission:
force-valid tokens enter force runs; contact fallback runs exclude those tokens.
Targets with object index -1 have no target run. Compatibility reconstruction
must preserve the complete original training-run partition. Object identity
continues to join instance to asset to geometry, never by numeric label alone.

The migration checks range bounds, source identity, video header counts,
real consumer reads, all label training-run partitions, and split intervals
before publication. Header checks do not imply full video decoding or GPU
training validation. Publication uses journaled NFS renames and retains the
entire previous dataset; old single-info rollback tools cannot roll this back.


Optional source sidecars whose entire payload tree is absent may be explicitly
disabled before migration, retaining original metadata in the external repair
backup. `manifest.unavailable_modalities` records `available=false` and its
reason; compatibility views preserve those explicit declarations. Missing
tactile, body-tracking or object payloads do not become zero-valued labels.
Historical prediction caches are not declared as installed coverage.


### Static object registry: object_assets_v1 (2026-10-21)

Processed object assets may use `assets/objects/assets.json` plus
`assets/objects/meshes.bin`. The manifest's `object_scene.asset_registry_format`
is `object_assets_v1`. `assets_path`, `geometries_path`, and optional
`articulation.parts_path` resolve to sections `assets`, `geometries`, and
`parts` in the JSON document's `tables` object. `mesh_store_path` resolves to
the unchanged little-endian float32 vertices/int32 triangle binary store.
Original Arrow schemas are retained for exact typed reconstruction; nonfinite
scientific values use explicit `__object_assets_float__` JSON tags.

`core.geometry.object_assets.read_asset_table` reads both the JSON registry and legacy
Parquet tables. Asset identity remains the exact string
`object_asset_id -> object_geometry_id` join. Dynamic per-frame facts,
instance membership, relationship groups and state tables remain Parquet.
This storage change does not alter any tactile/force labels or mesh arrays.


### Unified handcrop geometry (2026-10-21)

Processed `crop_geometry` payloads may declare
`context_format: embedded_handcrop_context_v1` in the manifest and
`beingh.handcrop.context=embedded_handcrop_context_v1` in their Arrow metadata.
They contain the existing nine v3 geometry/mask fields plus `crop_type`
(string), `width`, `height`, and `stream_num_frames` (int64). These fields
are joined exactly by episode/view/target and stored with every geometry row.
Independent `handcrop_context` payloads, modality declarations and range
indices are removed after validation. Pixel/normalized coordinate conventions,
frame IDs, physical row order, row-group boundaries and handcrop video offsets
remain unchanged. Legacy nine-field v3 geometry remains readable.
