# 复现、来源与归档边界

## 来源身份

- 主机：`ecs-36477934-002`，2026-09-11只读采集；当次快照8张A100 PCIe 40GB全部0MiB/0%，无GPU计算进程。
- 主实验：`/data/experiments/value-mixed-takeover-20260907/runs/piperx-vision-mixed1to1-c150-z15337-fixed1500-seed1000`。
- 原有最后checkpoint：`checkpoint-001500.pt`；250/500/750/1000/1250/1500全部仍在源机器，单个约4.8GiB。
- Evo-RL reference没有`.git`；runner记录的来源commit为`6f2db449a21e1bac750b996f2e27cac6739aa63f`，逐文件SHA才是本次工作副本的精确身份。
- 官方LeRobot源码commit：`7e241bd630a3719a56157a497ce5d08f244784f1`。实验中的`code/lerobot-v0.6.1-official`实际链接到`/data/src/lerobot-v0.6.1`，本归档将其源文件展开保存。
- 第三方代码沿用各目录内LICENSE、版权声明，不重新声明原创。

## 环境

`ENVIRONMENT.json`记录实际venv的`pyvenv.cfg`和安装包METADATA中Name/Version；`requirements-observed.txt`是便于检索的展开版本。它不是跨平台通用锁文件，也不含轮子SHA。

实际训练环境：Python3.12.3，torch2.10.0+cu128，torchvision0.25.0+cu128，transformers5.3.0，accelerate1.13.0，LeRobot0.6.1。CUDA wheel来源和安装顺序见原`value-demo3-20260906/setup_value_env.sh`。该脚本的默认目录是历史路径，使用前应明确部署目标。

训练helper将Evo-RL reference的values/utils加入官方LeRobot模块搜索路径，并兼容processor类型位置；这就是两套源码都需保留的原因。不要随意用当前仓库main的文件替代reference后继续旧checkpoint，因为恢复校验使用具体代码哈希。

## 恢复后仍需要什么

1. 先用`verify_snapshot.py`校验归档；用`restore_snapshot.py`恢复到一个新目录。工具自动解压JSON并还原文件名、普通文件权限和可用的文件链接内容。不会启动进程、连接服务器或获取模型。
2. 原代码固定引用`/data/experiments/...`、`/data/datasets/...`、`/data/cache/...`。恢复到其他目录不自动改写源码；未来在隔离环境部署时需规划对应挂载/路径。改写会改变原来的代码哈希与恢复约束。
3. 训练需要原始Demo/HIL数据和SigLIP/Gemma资产；数据来源、split、视频映射、SHA校验、成功来源均在manifest及原获取脚本中。旧setup/download脚本仍含网络/写入操作，归档本身不执行。
4. 精简模型还需要外部`assets/initial-seed1000.safetensors`；保存的`assets/export_manifest.json`和run protocol含既有SHA。`export_and_validate.py`展示如何从旧模型导出并做等价性验证，但它还依赖旧checkpoint进行后半段验证；不能把它当作无依赖的一步安装器。
5. 逐checkpoint推理需要源checkpoint；完整环境、GPU、视频解码库与数据齐备后才可重跑。本文不表示本次重新跑过GPU训练或推理。
6. 原`runs/`含完成标记和旧日志；新实验应选择新的输出目录，避免把历史状态当作新实验状态。历史队列、停止控制器、传输和上传脚本用于审查，不应因文件存在而自动执行。

## 收录与排除

四个实验树和官方依赖源码共登记7,355个文件项；5,362个实际文件收录。逐帧JSON/JSONL、Parquet标注/元数据、CSV、报告图、日志、源码备份、shell、配置、测试都保留。压缩只改变存储编码，不改变恢复字节。

大权重`.pt`/`.safetensors`、`.mp4`、tokenizer`.model`、`.pub`、`.lock`、`.pyc`为外部项。小型初始化RNG safetensors保留。排除完整venv、缓存、Git对象库和`__pycache__`目录。未下载约934GB的大产物，未读取/收录SSH密码或私钥。对外部大产物未重新计算全量SHA；已有实验记录中的SHA原样保留，清单仅对实际收录内容提供新计算的SHA。

文件软链接记录原链接目标；指向已收录文件的链接由恢复工具物化。指向未收录产物或源树外的链接仍需要外部依赖。目录链接官方LeRobot已展开。`FILE_MANIFEST.json`的`repository_path`是归档存储位置，`path`是恢复后相对于`/data/experiments`的原路径。

部分上游媒体文件本来就是 Git LFS 指针。它们通过JSON容器保存原指针文本和SHA，恢复工具还原指针字节；指针所引用的实体不在本归档中。这样既保存源目录的实际内容，也不向目标GitHub仓库推送缺少实体的LFS引用。

## 已知缺项：checkpoint比较脚本

原`REVIEW.md`写有`results/checkpoint_comparison.json`和`compare_checkpoints.py`。实际A100上结果位于混合实验根目录`checkpoint_comparison.json`；归档保留这个实际路径。搜索可访问的相关实验、源代码和用户目录没有找到该比较脚本，没有用推测实现冒充原文件。

已有六个checkpoint的metrics、advantage records、summary、预测、最终比较JSON均收录，可以检查数值并重新分析。报告中的匹配规则记录在`ALGORITHM.md`；缺失脚本的精确随机实现和绘图代码无法从报告保证恢复。

旧relay脚本中的`/home/zhaobo/value_relay_nonhf_v3_20260907.py`、`value_relay_transfer_v2_20260907.py`是另一台源机器上的helper路径；本次只访问A100，未连接该机器。对应A100 relay脚本中的send/receive实现已保留。

## 本次验证

本地验证原始和压缩存储的SHA、逐文件大小，并使用AST解析所有1,337个Python文件；不导入实验模块，不执行其顶层代码。另对恢复后的文件做独立哈希核对。新工具仅是归档校验/恢复辅助，不修改原算法。
