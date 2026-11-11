# 本次未超过Baseline的原因与独立修改版

## 1. 先核对真实结果，不能把“比失败版本好”当涨点

本次正式权重是`baseline_base_head_output_center_safe_joint_bs16_30ep_20260930`的Epoch30，run_id100130。700/700测试和评估完成。官方指标函数计算：

| 指标 | 本地官方E300复现 | 本次FULL | 差值（百分点） |
| --- | ---: | ---: | ---: |
| AUC | 62.73 | 62.71 | -0.02 |
| OP50 | 73.73 | 73.46 | -0.27 |
| OP75 | 58.90 | 58.96 | +0.06 |
| Precision | 66.21 | 66.18 | -0.03 |
| Norm Precision | 80.28 | 79.97 | -0.31 |

从未四舍五入的逐序列重算值看，AUC差为-0.01215百分点；表格各数先显示两位小数后相减为-0.02。两者不矛盾。439条序列正增益、253条负增益、8条相同，序列差中位数+0.173个百分点，但宏平均仍略负。不得用正增益序列数量替代最终指标。

## 2. 已确认与尚不能断言的原因

### 完整已存结果诊断

有效跟踪帧450073（排除初始化和无效/不可见GT）。以官方结果为参照分桶，原IoU>=.75的帧中，本次平均IoU差-0.01544；原IoU .50~.75的帧为-0.01854。原IoU低的帧则有改善。这说明综合收益抵消存在，不是所有困难样本都变得更差。

此差异包含Head微调、prompt历史及模块修正，不能单独归因于中心模块。分桶是对完整轨迹输出的结果分解，不是同状态因果对照。

原框宽高10%范围内的GT中心Oracle平均IoU提升空间：官方原框约0.03009，本次FULL约0.02830。这里是忽略图像裁剪的**静态几何诊断**，不能解释为可实现的AUC上限，更不能用GT选框部署。

### 小规模同轨迹核对

固定随机种子20261001，从全部700条等概率选32条，每条前128帧，3758个有效样本。在一次相同Head轨迹内同时读取native和refined输出，不重跑完整OFF测试。

- 模块浮点框平均IoU增益+0.000798。
- 按官方整数保存格式的平均IoU增益+0.002529。
- 同一范围内native相对已存官方轨迹平均IoU差-0.009467。
- 模块有益改变（IoU>+.01）20.30%，有害改变（IoU<-.01）7.48%。
- 平均残差x=.000476、y=.006512；平均原图中心移动x=.019、y=1.278像素；饱和率仅0.146%。

因此，**本次模块不是完全无效，也没有证据支持盲目加大限幅**。局部正贡献不足以抵消这一抽样中native路径的差距。该诊断只覆盖早期前缀，不是完整700条的模块消融，也不证明所有Head更新都有害。正向平均y偏移不自动等于错误：部分可能是合理修正或抵消整数截断偏差。浮点/整数差异提示必须分别记录，不能靠修改结果writer制造收益。

### 代码结构限制

旧输入为三路Head特征的选点值、局部均值、局部标准差。均值/标准差不保留邻域空间次序，整体输入并非严格反射不变，但缺少显式左右/上下比较。旧MLP允许在相似特征上产生共同方向偏置。

模块仅修正返回框中心，宽高保持原值，不反馈搜索状态/prompt。这减少模块引发漂移的风险，但不能修复native已经切错目标的历史，也不能解决尺寸错误。当前验证使用GT抖动裁剪，正式跟踪使用自身预测裁剪；验证局部IoU gain不等于全序列AUC。

## 3. 本次独立修改，不覆盖E30原模型

配置名：`baseline_base_head_directional_center_guard_bs16_30ep_20261001`。

### A. 保持Head联合训练，增加实际决策退化约束

训练期复制官方E300 Head为冻结教师，在同一个冻结Backbone输出上运行。额外只运行Head，不重复运行大Backbone。教师不进入优化器、不进入正式推理。

使用与测试一致的`C*T*Hann`选点和图像坐标框解码：

1. 几何约束：用教师GT IoU平方作为连续权重；当学生选中框GIoU低于教师时惩罚差值。GIoU在不相交时仍可提供梯度。
2. 排序约束：仅当学生在教师选点位置的实际框IoU高于学生当前选中框时，推动该位置排序超过当前赢家。不是无条件照抄教师argmax，不是整图KL响应蒸馏。
3. 学生更好时不惩罚；flag=0/1不加此VL决策约束，原Head训练仍继续。

权重显式配置为box=.5、rank=.25，这是初始实验设定，不是审计推导出的最优数值。该约束不保证长期闭环保持Baseline，特别是历史身份切换和自然轨迹分布差距仍需正式结果验证。

### B. 中心修正改为方向有序的证据比较

保留288维原特征上下文，再在原框中心沿x/y各取正负四分之一宽高的特征，明确保留`+x/-x/+y/-y`顺序。三路Head共96维边侧特征；crop归一化坐标用`grid_sample(align_corners=False)`，训练和测试共用采样代码。

共享函数f对两侧证据作成对比较：

`delta_axis = .1 * tanh((f(context,pos,neg)-f(context,neg,pos))/2)`。

- 对称两侧证据相同，修正严格为0。
- 交换两侧证据，该轴残差严格反号。
- 末层零初始化，启动时严格保留native输出。
- 31808参数，宽高不修改，仍只输出细化、不反馈内部状态。
- 保留原2GIoU+5图像归一化L1及IoU/中心误差损害约束，不同时修改其损失系数。

这些代数性质只针对给定侧证据的交换，不宣称整个Backbone/跟踪器具有图像反射等变性。缺少身份信息、丢失目标等问题不会仅靠方向比较自动解决。

## 4. 严格保持的实验条件

官方E300重新初始化；不从本次E30续训新结构。Backbone冻结/eval，Head解冻，Head BN运行统计固定；模块梯度不跨入Head。

四训练源、比例、30000/轮、batch16、VL_RATIO=.44、GROUNDING_RATIO=.11、全部验证数据与规模不变。Head LR1e-6、模块LR1e-4、AdamW、weight_decay1e-4、分组梯度裁剪.1、EPOCH=T_MAX=30、每轮验证均不变。

新旧YAML差异只有三项：`MODEL.CENTER_OUTPUT.DESIGN`及两个`TRAIN.NATIVE_DECISION_*_WEIGHT`。旧ERC/RQC重排序仍关闭。**当前方法已是选点后定位细化，不再是最初“仅重排序、不移动框”的RQC，不能在论文里混用定义。**

## 5. 检查及当前状态

- CPU结构检查：零初始化等价、侧证据交换反号、同侧证据零修正、实际空间采样次序、GT/原框/特征梯度隔离通过。
- 合成监督检查：坏决策产生恢复排序和框梯度，教师无梯度，学生更好/相同不惩罚。
- 真实训练与全部5路验证loader检查通过；两个活跃训练batch的Head/模块梯度均有限非零；纯BBOX模块无梯度。
- 教师初始权重等于官方Head，检查结束仍完全相同；Backbone/Head BN统计不变。
- I/O回归：22帧零残差与native严格一致；24帧人为非零残差只改返回框，state/prompt严格一致；合成fixture被正式测试入口拒绝。
- 最终真实检查报告用`directional_center_real_check_verified_20261001.json`。首次检查发现Actor方法解析遮蔽监督入口，已在新文件内修复；没有执行任何训练更新。首次检查报告保留但不作为最终通过凭据。
- 训练、测试、评估采用独立入口/YAML/variant/目录/文件名前缀。共享4个接入文件、旧checkpoint与旧正式结果没有覆盖。
- 优化器更新0步；**未启动新训练**。这些检查证明可运行，不证明可以涨点。

## 6. 确认后的固定实验

服务器项目根：`/root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated`。

确认后从官方E300启动固定30轮，先检查前两轮数值/梯度、Teacher-vs-Student决策退化率、中心有害修正与方向分布，再正常跑满。不开新的三轮筛选，不按TNL2K个别视频调阈值，不自动启动额外OFF。

```bash
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uvltrack
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python change/rqc_dev_v3/train_directional_center_joint_20261001.py --train
```

此处命令只是记录，当前未执行。新目录`change/directional_center_joint_training_20261001/<新配置名>/`若存在，入口拒绝覆盖。每轮文件`DirectionalCenterJoint_epNNNN.pth.tar`，存训练后Head和新模块，以及优化器/调度器/随机状态；冻结官方Backbone由明确E300源提供。训练期教师不随学生checkpoint改变。

正式测试仍复用官方dataset/runner/writer/指标，但必须用独立新tracker加载对应Head+模块，4进程单卡；新run_id和独立评估目录，禁止把旧CenterJoint权重加载到新结构。

下一次先固定E30 FULL TNL2K判断综合结果；有效后固定同一权重验证LaSOT/OTB99。TNL2K已被反复用于开发，结果应如实报告，不能当作未经选择的泛化证据。最终论文仍需要合理消融，但现在不新增完整OFF耗时实验。

## 7. 用户确认后的启动记录

用户于2026-10-01明确要求启动训练。再次核对最终检查报告及全部源码/YAML/官方权重SHA、单卡和不存在的新输出目录后，12:17:50（北京时间）后台启动。PID10351，实际Python为`/root/miniconda3/envs/uvltrack/bin/python`，无DDP。

日志：`terminal_logs/train_directional_center_guard_30ep_20261001_121750.log`。12:18:32已经Epoch1 step100；Head/模块梯度有限非零、学习率1e-6/1e-4，模块残差已离开0，新Head决策约束在step50/100出现非零几何regret。此时没有完成首轮验证，不能据此宣布涨点或稳定泛化。GPU采样81%、显存2726MiB。

本次只启动固定30轮训练，不添加定时监控、不自动启动正式测试。此前“尚未训练”的检查章节记录的是启动前事实，以本节为最新状态。

## 8. 训练结束与用户授权测试

2026-10-01 16:26:52已保存正式Epoch30，训练打印`TRAIN_COMPLETE`，训练进程10351消失。训练循环耗时14930秒（4小时8分50秒）；日志没有NaN、非有限梯度、OOM或Traceback。Epoch30 VL验证中心IoU gain=.00774375，有益比例25.039%、有害8.109%、严重有害0；仍不构成全序列涨点证据。

用户明确要求结束后进行下一步，故16:42:23启动Epoch30 TNL2K NLBBOX FULL，**不是未经授权自动测试**。新run_id101230，4个worker、1张GPU；只运行FULL，700条完成且完整性检查通过后自动使用官方指标函数评估。

正式checkpoint SHA256：`1524962167d8f5b60a33b29960feae47f82b8e2a17869fc628e2f543978267bd`。启动器PID24124，测试主进程24137；实测工作进程24146/24147/24148/24149，Python均来自uvltrack虚拟环境。16:43 GPU采样73%，显存4360MiB，已经产生首批结果，无错误。

- 启动脚本：`change/rqc_dev_v3/launch_directional_center_e30_tnl2k_full_20261001.sh`。
- 启动日志：`terminal_logs/launch_directional_center_e30_tnl2k_FULL_run101230_20261001.log`。
- 测试日志：`terminal_logs/test_directional_center_ep30_tnl2k_FULL_run101230_20261001.log`。
- 评估日志：`terminal_logs/eval_directional_center_ep30_tnl2k_FULL_run101230_20261001.log`。
- 预测：`test/tracking_results/uvltrack/baseline_base_head_directional_center_guard_bs16_30ep_20261001_101230/tnl2k_NLBBOX_030`。
- 指标：`change/directional_center_joint_evaluation_20261001/baseline_base_head_directional_center_guard_bs16_30ep_20261001_tnl2k_e0030_run101230`。

启动前核对了官方源权重、新权重variant/epoch/配置与源码指纹、30轮统计和不存在的目标目录。测试同时加载**训练后的Head与方向模块**，没有误用官方Head配新模块。所有旧正式测试与评估目录保留；不创建定时监控。
