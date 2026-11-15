# 标签格式

[English](label_format.md) | [简体中文](label_format.zh-CN.md)

## 存储与索引

`<dataset>_lerobot/meta/info.json` 指向活动 prepared manifest。
整数范围将 episode、模态和逻辑行映射到文件及偏移，采用半开区间。
Episode 可跨文件和 row group，读取端须遵守行偏移、行数及各流的源帧映射。

规范化容器声明 `format_version=processed_dataset_v1`，包含 `meta/`、`data/`、`assets/`。
元数据存放 manifest、表和索引，tactile 与 force 是 `data/` 下的模态。
时间戳只作辅助，不替代 solver 行、transition 和 token 索引。
物体身份按字符串 ID 进行实例 → 资产 → 几何关联。

## 触觉标签表

| 组件 | 行单位 | 内容 |
|---|---|---|
| `samples` | Solver 行 | `contact`、`signed_distance` |
| `transitions` | Solver 状态行 | 世界系顶点力、接触/环境 wrench、目标 COM 位姿与速度 |
| `timeline` | Solver 行 | `solver_row`、`solver_time_s`、`source_position`、`tactile_flags`、`force_valid`、`target_object`、`closing_solver_row` |

紧凑 manifest 声明 `row_metadata: tactile.timeline`，reader 从 timeline 还原逻辑字段。
三表按 episode、solver 行对齐；稀疏静帧可声明 `solver_index_column: source_position`。

- `contact` 为 uint8，`signed_distance` 为 float16 米，截断至 ±0.25 m。
  每行先排列左手 778 顶点，再排列右手 778 顶点。
- `tactile_flags` 的 bit 0/1 表示左右手 presence，bit 2/3 表示接触/距离有效性；
  `force_valid` 单独控制力监督。
- `source_position` 是源帧坐标，可为小数。静帧可有独立对象槽位、重力及空 solver 时间。
- 动态 `closing_solver_row` 仅在下一状态采用相同目标参考系时存在。
  直接力监督可用最后有效行，rollout 还需要连续时间与结束状态。
- 力和 wrench 采用求解器参考质量约定。参考质量、COM、惯量及 `object_ids`
  与目标槽位对齐；关节数据另存广义 wrench 和关节状态。

Episode context 记录拓扑、频率、物理量及摘要。`tactile_sha1` 按
`solver_sample`、`source_position`、`flags`、`contact`、`signed_distance` 顺序
校验逻辑数组，不受分片方式影响。`force_sha1` 覆盖声明的力、状态及附加物理字段。

## 几何与裁剪

统一通过 `core/geometry/mano_layer.py` 解码。读取顺序为已安装 `mano_contract`
→ `mano_source_basis` → `MANO_BASIS_DECISIONS` 指定的旧数据审核记录。
不按数据集名称推断基底，也不在消费者重复翻转。规范参数为
`[trans3, full_axis_angle48, betas10]`，单位米/弧度，采用
`flat_hand_mean=True`、`use_pca=False`、`center_idx=None`。

迁移记录模型指纹、来源证据和 `geometry_revision`。所有有效帧参数与顶点须有限，
逐顶点最大误差不超过 1 mm；已记录的 HIC 例外为 2 mm。无效或超门槛分片不得安装。
`hmr.mano=false`、`mesh_vertices=true` 时优先读取存储网格，不解码占位 MANO 参数。

当前裁剪框使用观测平面的归一化坐标：`crop_box_xyxy_orig`、`hand_bbox_xywh_orig`；
`hand_bbox_xywh` 仍为裁剪像素。负值和图像外窗口用于补边，不应截断。
通过 `frame_intrinsics` 从观测相机与框推导 K，保留声明的输出尺寸和图像平面。
旧 raw-mirror 坐标与存储 K 仍可读取，原生鱼眼 K 不能替代校正 K。

几何包含 episode/view/target/frame 身份及独立的 `frame_valid`、`crop_valid`。
嵌入 context 还包含 `crop_type`、`width`、`height`、`stream_num_frames`。
Packed 偏移按声明映射恢复，不把缺失索引区间压成累计行数。

## 物理参考与发布

多体参考在训练前离线恢复到 `rigid_reference_v1`，不在 loader 中计算。
每个 episode 有 `bodies.parquet` 和绑定源哈希、恢复版本的 manifest，记录逐刚体
COM 状态、重力、wrench、顶点归属（int16，-1 表示无归属）及有效性。
缺少动力学支撑的刚体保持 neutral，全部所需一致性检查通过后才发布 ready。

```bash
python scripts/labels/solve.py --help
python scripts/labels/publish.py --help
```

求解输出 canonical NPZ，发布生成目标槽位、时间轴和监督区间，无效力行清零。
`--install` 验证载荷后原子激活 prepared manifest，保留旧引用以便回滚。
完整容器迁移须按整套数据回滚，不能只恢复单个 info。
可选模态缺失时显式记录不可用，不生成零标签。

几何改变后，依赖标签标为 stale，训练须拒绝读取。新求解绑定 HMR 指纹和几何版本，
对账后发布器才能清除 stale。标签改变时重建相关统计和参考，安装前验证标签摘要及真实裁剪消费者。
