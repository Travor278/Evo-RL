# SFT + RL 消融完整性审计（2026-09-11）

**结论：没有完整实现或完成用户本次定义的 SFT + RL 实验。** 仓库已有SFT、干预数据重放/ACP、Value训练、离线Advantage和片段导出等实现；不能将这些组件或旧模型合并称为已完成的纯示范SFT+RL。此次仅审计与归档，未启动实验或修改原算法。

审计标准：只用允许的纯人工遥操示范，从同一SFT checkpoint继续；纯示范Value、训练集最大累计成本Z（不乘2）、gamma1/n50、全局Top10%且正值、保留20连续动作、D+D_high显式重采样、普通文本/flow matching、真实loss padding mask、匹配对照预算、最终真机协议。

## 逐项判定

| 要求 | 当前证据 | 判定 |
|---|---|---|
| 从同一SFT checkpoint续训 | 本地历史H100记录存在step50000 SFT及哈希；旧BC/ACP配置引用其兼容版，但没有本次SFT+RL的冻结配置与加载证明 | 候选存在，本次未接线/核验 |
| Value只使用纯示范 | 当前片段构建明确使用`mixed1to1-c150-z15337`的1500步模型；训练每卡demo4+hil4 | 不符合，不能复用 |
| 纯视觉201桶two-hot CE、期望V | `lean_model.py`和`train_mixed.py`已实现，无language/state | 组件符合，可复用结构 |
| 纯示范奖励与正确Z | 旧纯示范`Frames`使用`2*max(length)`；PiperX旧run实际Z14882。混合run Z15337且有150接管罚分 | 现有两类checkpoint均未通过本次归一化/来源要求 |
| 完整episode划分、CE/MAE | 旧Demo502/56划分与训练/留出指标存在 | 机制可复用，需先核对本次允许集及最终policy测试隔离 |
| 对齐Value预算和预先选型 | 历史有1500/4500/8000等不同预算，不能自动选择一个当论文对照 | 本次对应对照未锁定 |
| 完整轨迹n50、gamma1、终点0、截短 | `build_demo558_segments.py:34-36`及`evaluate_advantage.py:85-93`具备计算逻辑 | 算法组件符合，现有输入Value不符合 |
| 全局Top10%再A>0，扩展20/合并 | `build_demo558_segments.py:38-50`先全局取ceil(10%)再过滤正值，合并相邻/重叠区间 | 规则符合；旧导出不能作为本次D_high |
| 最终policy测试数据不参与筛选 | 当前export硬编码`range(558)`，无本次allowed-set/final-policy-test排除契约 | 未证明隔离，需补显式清单及拒绝检查 |
| 源区间/起点/分数/阈值/模型可追溯 | 现有导出plan、frames注释与报告保存这些信息 | 可复用 |
| 视频/action/state/时间戳/读取验证 | `build_demo558_segments.py`与`review_exported_demo558.py`有实际导出验证 | 旧产物通过，不代表新产物已通过 |
| D+D_high显式采样 | 仓库有base/HIL WeightedRandomSampler；旧组用base/attempt重放。未找到本次纯示范D_high的混合集、来源标记与sampler配置 | 未完整实现 |
| 普通任务文本、flow matching | BC模式可关闭ACP，Pi0.5已有flow loss；ACP路径会加标签，不可用于本组 | 有底层能力，需本次显式关闭并验证 |
| policy horizon保持对照、短片段mask进loss | Pi0.5 `forward`直接`losses.mean()`/`mean((1,2))`，没有读取`action_is_pad` | 明确不符合，必须补mask并验证 |
| 采样比/更新数/global batch/LR/参数范围对齐 | 找到旧attempt BC/ACP 50:50、20k、global64/BF16；那是另一种数据和目标，不等于本组已批准参数 | 仅候选证据，不自行套用 |
| 本次policy日志/最终ckpt/加载推理 | 未找到符合上述纯示范契约的运行产物 | 未完成 |
| 论文协议真机SR/TP、各阶段成功/尝试数 | 已有离线MAE/flow报告明确不代表机器人SR；没有本次SFT+RL真机结果 | 未完成，论文该行不能补为已验证 |

## 两个容易误判的点

### 纯示范数据导出不代表纯示范Value

`build_demo558_segments.py`只导出Demo558的动作，但它检查`step==1500`、固定`scale==15337`，读取混合Value的预测。这仍违反“不能使用干预数据训练的Value”的实验边界。必须从合规Value重新推理完整源轨迹与重新导出。

### Dataset产生padding标记不代表loss使用了它

LeRobot dataset按episode边界截取/补齐，并生成`action_is_pad`。审计的main Pi0.5 `src/lerobot/policies/pi05/modeling_pi05.py:1248-1283`和归档reference的相应forward只对action维裁切后平均全部时间步，没有按有效动作掩码归一化。20帧片段用于50输出horizon时，额外补齐部分会进入监督。旧attempt sampler要求完整合法50帧chunk，能避开旧组的短chunk，但直接复用会丢弃本组短片段，不能替代mask实现。

## 已找到的对照候选，不是缺省实验参数

审计证据副本放在`audit_evidence/`：

- 原SFT记录：8H100、global64、全参数FP32、50k，LR2.5e-5，warmup1200，decay50k到2.5e-6；最终模型SHA`d85c7cd84060a924b6ef10d055491c500c5714a21ddbb49e1dbdc828b7a74147`。
- 旧attempt-aware data-only/ACP配置：同一个step50000兼容SFT路径、20k、global64、BF16、50%base+50%attempt、chunk50。BC关闭ACP，ACP开启并dropout0.30。
- launcher从checkpoint/policy preset继承一些参数，单靠YAML和当前默认值不能确认有效LR、调度、可训练范围；必须读取正式对照run的最终train_config/provenance。
- 旧50:50是base/干预attempt比例，不能未经核对解释为D/D_high比例。旧20k更新也不能自动成为本实验预算。

本次待核对字段完整列在`sft_rl_required_config.json`。未知项为null，没有猜测采样比、更新预算、输出目录或Value checkpoint。

## 完成此消融还需要的工作

1. 锁定论文对应对照、SFT checkpoint、允许示范版本/episode清单、最终policy测试清单、有效训练配置和固定预算。
2. 复用纯视觉结构，构建无接管罚分、Z=max_train(T)的纯示范Value训练配置；超范围留出标签采用明确报错/审查规则，不静默clip；重新训练或核验真正符合要求的历史checkpoint。
3. 合规Value全轨迹推理，按现有正确Top10%/A>0/n50/20窗口逻辑导出；加入最终policy测试排除检查。
4. 接通D+D_high的显式采样与可追溯来源，修复并验证Pi0.5时间padding mask确实参与flow loss。验证不跨片段、horizon与对照一致、不添加ACP文本。
5. 冻结配置后再运行policy续训、加载/推理验证；最后按论文真机协议取得SR、TP及阶段计数。

这是一份完整性审计，不是宣称上述缺项已实现的计划验收记录。
