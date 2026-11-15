# 开发指南

[English](development.md) | [简体中文](development.zh-CN.md)

## 架构

| 目录 | 职责 |
|---|---|
| `core/data`、`core/io` | Prepared manifest、表、索引及视频/文件 IO |
| `core/geometry` | MANO、网格、相机、投影和质量属性 |
| `core/physics` | 数值动力学、接触求解和 wrench 聚合 |
| `core/tactile_labels` | 标签存储、读写和手部裁剪读取 |
| `core/visualization` | 共享场景渲染和色图 |
| `training/` | 模型、学习数据、训练、推理、评估及结果展示 |
| `label_generation/` | 离线求解适配、调度、审计和发布 |
| `scripts/` | 数据、模型操作及硬件配方 |
| `tools/` | 开发检查、打包和工作区维护 |

Core 不导入 training、label_generation、scripts 或 tools。
训练与标签生成共用 core，通过标签文件交换数据，不互相导入。
数值物理接收显式状态和参数，读取层不依赖渲染。共享实现放在库中，不放在 CLI。

学习窗口、增强和监督资格位于 `training/data/`；模型网络及专属运行适配位于
`training/tactohand_wrench/`，HOPE 是独立任务。移动代码时保留模型路径、
checkpoint 键和已保存 Hydra target 的兼容性。

看板通过 `/api/server` 管理服务端配置，上传推理请求使用服务端选定的 checkpoint 和设备。
本地客户端分别保存各浏览器的连接目标，并代理视频上传、结果读取及分段视频请求。

## 操作与检查

在仓库根目录运行命令。真实数据检查放在 `scripts/data_checks/`，开发回归放在
`tools/checks/`，预设放在 `scripts/<workflow>/recipes/`。
配方选择参数并调用可复用操作。平台入口必须等待真实任务结束并返回退出码；
长任务使用独立日志、增量输出和完成标记。

```bash
tools/run_checks.sh --quick
python3 -S tools/checks/test_inference_dashboard.py
python tools/checks/test_inference_visibility.py
python tools/checks/test_inference_server.py
```

`--quick` 检查 shell、跟踪文件卫生和依赖边界，`--full` 增加 launcher 契约和 pyright。
预测、资源、rollout 检查单独运行，按改动选择相关项。
涉及数据时先用小子集验证；几何或裁剪改变后，检查实际消费者。

## 打包

`packaging/public_files.toml` 控制 wheel 和源码包范围。
源码包包含中英文文档，本地数据、checkpoint 和第三方模型文件单独准备。

```bash
python tools/packaging/build_public_source.py --output /tmp/tactohand-source.zip
python tools/checks/check_distribution.py \
  --output artifacts/release_checks/distribution.json \
  --log-dir logs/release_checks/distribution
```

分发检查构建源码包、wheel、sdist，比对重建库内容，并在仓库外用合成 CPU 输入运行安装包。
检查复用现有依赖，新环境安装和真实权重评估另行验证。
报告增量保存，全部成功后写 `.done`。

打包清单不改变 Git 可见性，也不删除研究历史。公开更新只包含选定源码，不合入研究历史。
分发前审核代码、统计资源、依赖和模型的使用权；模型文件不分发时仍保留来源元数据。

## 贡献

提交保持聚焦，标题采用范围加祈使句，例如 `training: fix force masking`。
PR 说明目的、受影响数据集和运行命令。Schema 或数据契约改变时更新
[标签格式](label_format.zh-CN.md)，中英文说明保持一致。

退役配方和调查记录保留在 Git 历史。当前操作写入[使用指南](usage.zh-CN.md)，
读取约定写入[数据集](datasets.zh-CN.md)。
