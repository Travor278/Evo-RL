# A100 Value / Advantage 完整实验代码归档

2026-09-11 从 `ecs-36477934-002` 只读采集。这里保存训练实际使用的代码、历史版本、依赖源码、配置、划分、日志、逐帧预测、评估和筛选记录；未在服务器执行训练、推理、队列或修改操作。

**本归档不等于已完成论文中的纯示范 SFT + RL。** 按最新协议逐项检查发现混合Value来源、旧纯示范Z乘2、D_high采样接线和policy padding loss等缺项。见 [SFT + RL完整性审计](SFT_RL_AUDIT.md) 与 [待核对配置](sft_rl_required_config.json)。

## 从哪里开始

| 目录（均在 `snapshot/` 下） | 内容 |
|---|---|
| `value-demo3-20260906/` | 最初的多任务 value 实验、90/10 整轨迹划分、数据审计、训练环境安装脚本、报告、模型配置与 tokenizer；`code/` 包含实际 Evo-RL reference 源码及官方 LeRobot 源码 |
| `value-ablation-20260906/` | 视觉/状态等消融、固定步数训练、state probe、结果分析、历史控制器 |
| `value-vision-schedules-20260907/` | 精简纯视觉 SigLIP 模型、等价性导出检查、学习率实验、跨任务实验、初始化配置和 RNG |
| `value-mixed-takeover-20260907/` | Demo/HIL 混合 value、接管罚分、advantage 诊断、六 checkpoint 比较结果、全 Demo558 推理与片段导出、视频渲染、传输与发布脚本 |

建议先读 [算法与调用链](ALGORITHM.md)、[复现与归档边界](REPRODUCIBILITY.md)，再看 [原实验最终报告](snapshot/value-mixed-takeover-20260907/REVIEW.md)。原始 `PROTOCOL.md` 含历史运行状态；最新完成情况应以 `REVIEW.md`、checkpoint 元数据和实际日志为准。

## 主要入口

以下路径相对于 `snapshot/value-mixed-takeover-20260907/`：

| 文件 | 作用 |
|---|---|
| `mixed_data.py`、`manifest.json.gz`、`reward_config.json` | 接管奖励、MC 回报、数据池、帧采样、划分与标注来源 |
| `preflight.py` | 数据和边界检查 |
| `train_mixed.py`、`run_experiment.py` | DDP 训练、保存恢复、密集评估和历史执行队列 |
| `evaluate_advantage.py`、`review_checkpoints.py` | 多步 advantage 分解、整 episode bootstrap、六 checkpoint 评估 |
| `infer_demo_train502.py`、`infer_demo_heldout_for_video.py`、`infer_demo_global30.py` | Demo 推理与历史筛选分析 |
| `build_demo558_segments.py`、`queue_demo558_export.py` | 全558示范 A50 全局前10%且正值起点，扩展20帧并合并 |
| `review_exported_demo558.py` | 导出片段与原始数据逐项核查 |
| `render_*.py` | 全部历史视频可视化与不同筛选窗口对比 |
| `transfer*.py`、`relay*.py`、`direct*.py`、`hf_acquire_v3.py`、`upload_demo558_mirror.py` | 数据获取、校验、历史传输与发布实现 |
| `runs/`、`logs/`、`checkpoint_comparison.json` | 每次实验的配置、指标、逐帧预测和诊断记录 |

这些入口是原始实验源码，包含固定服务器路径；归档不是一个可直接无配置运行的新训练包。历史队列、停止控制器和上传脚本均保留用于审查，本次没有执行。

## 文件完整性

`FILE_MANIFEST.json` 逐项记录原路径、大小、时间、权限、链接目标和归档去向；收录文件同时记录原字节 SHA-256 与存储 SHA-256。共收录 **5,362 个原始文件、1,337 个 Python 文件**，包含第三方依赖源码与测试。

超过256KiB的 JSON/JSONL 使用确定性 gzip 无损压缩。嵌套 `.gitattributes` 改存为 `.gitattributes.source`，防止归档内容触发额外 Git LFS 或换行转换；恢复工具会还原原文件名。算法源码没有重写。

上游源码中的 Git LFS 指针另存为 `.lfs-pointer.json` 元数据封装，避免引用当前仓库不存在的 LFS 对象；恢复后仍是原指针字节，不是对应图片或权重实体。

```bash
python examples/a100_value_advantage_snapshot/verify_snapshot.py
python examples/a100_value_advantage_snapshot/restore_snapshot.py /new/empty/path/a100-experiments
```

第一条仅做本地哈希和 Python 语法检查；第二条只把文件恢复到一个**尚不存在的新目录**，不运行任何实验，不覆盖已有目录。Windows 长路径由工具处理。已归档字节约321MB，Git传输会进一步压缩。

## 外部产物与明确缺项

模型大权重、训练 checkpoint、原始/生成视频、tokenizer `.model`、公钥和运行锁未放进 Git；其路径与大小在清单中。环境缓存、`.git`、`__pycache__` 和完整 venv 也不复制。模型配置、small RNG、依赖版本、原有权重哈希与获取/导出代码均保留。这是完整代码及实验记录归档，不是约934GB训练产物的镜像。

原报告提到的 `compare_checkpoints.py` 未在可访问的相关目录找到；已有 `checkpoint_comparison.json` 和各 checkpoint 的原始预测/评估结果全部保留，没有伪造该脚本。详细说明见 [复现边界](REPRODUCIBILITY.md)。
