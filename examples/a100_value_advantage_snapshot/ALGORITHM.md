# 算法与源码调用链

## 1. 混合轨迹奖励与 MC value 标签

`snapshot/value-mixed-takeover-20260907/mixed_data.py:48-59`：

```python
r = np.full(n, -1., dtype=np.float64)
r[-1] = 0
events[:-1] = (~mask[:-1]) & mask[1:]
events[-2] = False
r[events] -= 150
g = np.cumsum(r[::-1])[::-1].copy()
```

`mask` 表示人工控制。一次自主→人工切换，在最后自主帧计罚；初始人工控制不罚，进入终点的切换不罚。gamma=1。标签 `y_t=G_t/Z`，其中 `Z=15337` 是训练最大累计成本，倍率1。等价于负的“剩余转移步数+150×未来接管次数”除以15337。成功终点为0。

全部轨迹按成功处理；HF66条的成功来自元数据，4090A75条成功来自当时用户确认，并非原始成功字段。此标签来源信息在 manifest 中保留。

Demo558：502训练/56留出；HIL141：127训练/14留出，按来源轨迹身份划分，不按帧切分。Demo/HIL每步精确1:1；池内按帧均匀有放回抽样，长轨迹自然贡献更多帧。每卡4+4，4卡全局32。

历史 demo-only 的归一化是 `2*max_train_episode_length`；不能与本次 `Z=15337` 混用，也不能直接跨实验比较归一化 MAE。

## 2. 分布式 value 回归

调用链：`train_mixed.py` → `train_lean_value.py` → `lean_model.LeanVisionValue`；two-hot 函数来自 `value-demo3-20260906/code/Evo-RL-reference/src/lerobot/values/pistar06/modeling_pistar06.py`。

输入为多相机当前图像，无语言、state、action。图像缩放384×384，以0.5均值/标准差归一化。SigLIP pooler输出1152维，投影512维，经有效相机掩码平均、LayerNorm和MLP输出201个logits。模型429,182,729参数，视觉编码器参与训练。

201个桶中心等距覆盖[-1,0]，间隔0.005。连续标签按距离分配给相邻两桶，用交叉熵拟合；预测 `V=sum(softmax(logits)*centers)`。这是监督 MC 回报回归，没有 TD target network 或 Q 网络。

训练1500步，seed1000；AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=1e-5；warmup200，峰值5e-5，余弦至1e-6；梯度裁剪10。FP32参数、BF16 autocast、gradient checkpointing、4卡DDP。每250步保存，逐rank RNG与优化器状态随 checkpoint 保存。恢复时校验代码、manifest、初始化等哈希。

`export_and_validate.py` 保存旧模型的实际使用张量，构建精简 SigLIP 模型，检查原始/精简模型实际图像 logits 和 FP32 梯度等价。`assets/initial-rng.safetensors` 已收录；两个约1.7GB模型文件作为外部产物登记。

## 3. Advantage 是离线 n-step 残差

`evaluate_advantage.py:85-93`，e=min(t+n,T)：

```text
time_value_delta = -(e-t)/Z + V(o_e) - V(o_t)
explicit_penalty = -150 * takeover_count[t:e] / Z
full_advantage   = time_value_delta + explicit_penalty
```

终点V强制0。n=15/30/50，主窗口50帧即30FPS下1.67秒。每5帧采样的自主帧网格用于ROC/AP；额外密集接管前窗口只用于事件图与负值比例，不混入自然发生率分类统计。置信区间用300次整episode bootstrap。

该分数使用未来图像和已发生的接管事件，是离线评分。显式罚分本身由事件标签定义；去掉它仍不是在线探测器。若V精确等于每条实现轨迹的MC回报，残差会望远镜相消为0。代码直接检查这一恒等式。

## 4. Demo片段筛选

`build_demo558_segments.py:24-70` 读取1500步模型对所有558条示范的预测。纯示范没有接管罚分，因此 `A50=-(e-t)/15337+V(e)-V(t)`。排除终点，在所有示范帧之间做全局稳定排序，先取ceil(10%×帧数)，再要求A>0。每个入选起点覆盖 `[t,t+20)`，重叠/相邻片段合并，不连续段分成不同输出episode。

保留原动作/state和来源注释；三相机视频切片、重编码、PTS与完整解码验证；用官方LeRobot重新计算统计与分位数。片段结尾不是任务成功终点；片段内部帧未必前10%。若策略仅应从入选anchor训练，需要消费选择标注，不能把所有片段帧都当作入选anchor。

## 5. 已有结果与限制

六个checkpoint均有value/advantage结果。1000步留出MAE约0.05049，是该1500步schedule中的候选；不等价于重新设置总步数1000。1500步完整A50的AUROC约0.6529，时间+V差约0.5317，非接管窗口也有51.5%负A。报告未证明可靠动作识别或下游RL策略收益。

报告中的匹配分析：同episode、完整50帧窗口、进度距离≤episode长度5%、两窗口不重叠，按正例时间顺序选最近且未使用的控制起点；相等分数计0.5；1000次episode bootstrap。该分析的结果已保留，但报告所称 `compare_checkpoints.py` 原文件没有找到，不能声称可逐字节复现其随机细节。
