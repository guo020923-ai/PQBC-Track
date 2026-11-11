# 中心细化受限联合微调：下一阶段主实验

2026-09-30补充：第一次训练因裁剪side元数据float32边界错误停止；修复详见`CENTER_JOINT_METADATA_FIX_20260930.md`。新默认输出目录为`change/center_joint_training_exact_metadata_20260930`，旧失败目录保留。以下未训练检查结果为首次启动前记录，不等于最新训练状态。

## 决策和边界

从官方Epoch300重新初始化，冻结Backbone，Head和中心细化模块同时训练。不是先训Head再训模块，不恢复旧ERC/RQC候选重排序。保留已经准备好的冻结Head版本，不覆盖其文件、检查报告或旧实验结果。

采用联合版是为了验证最终综合收益，并不是已经证明联合训练优于冻结。BQ冻结Head的正式TNL2K AUC61.95，JQ联合版62.15，二者都低于本地baseline62.73，且模块设计不同，不能作为冻结策略的单因素比较。自然轨迹审计主要发现质量高估、有害误选和排序空间受原响应限制，未证明Head梯度冲突是主要掉点原因。

当前结构已经改变为**响应选点后的中心输出细化**，不再是“仅重排序、不移动框”。输出细化不要求冻结Head：联合版的内部状态来自训练后Head的原始预测，新模块仍只修改返回框，不反馈内部搜索状态或prompt。不能再声称内部轨迹等于官方Epoch300；正确表述是与**同一训练后Head的native路径**保持一致。

## 同时训练、梯度职责分开

- Backbone：参数冻结、eval。
- Head：15,718,505个参数可训练，原始UVLTrack损失不变。包含官方GIoU/L1/分类/aux/对比项，不额外加入旧ERC蒸馏、锚定或peak preservation。
- Head的BatchNorm：running_mean/running_var/num_batches_tracked固定；BN的仿射参数仍可训练。Head其余模块在训练时处于train模式。
- CenterSafeRefiner：288维Head定位特征→64维MLP→2维中心残差，19,202参数；末层零初始化，中心限幅原框宽高10%，crop宽高不修改。
- 总损失：`L_original + L_center`。原始损失只训练Head；中心损失只训练新模块。捕获特征、原框、GT和离散Hann选点detach，不让新损失直接拉动原Head。
- Head的更新仍会改变后续batch的原框和特征，因此这里不是两个永远固定输入的独立模型，也不是完全端到端梯度耦合。
- 纯flag=0/1批次仍更新Head，但新模块梯度为None；其AdamW动量与weight decay不推进。
- 新损失仍采用2GIoU+5原图归一化xyxy L1，以及相对原框IoU/中心误差损害约束，各损害权重1。不改变当前限幅、结构或损失系数。

训练标注逆映射和实际测试解码分开，保留前一轮修正的半像素一致性；原增强流程不变。

## 超参数和数据

独立YAML：`experiments/uvltrack/baseline_base_head_output_center_safe_joint_bs16_30ep_20260930.yaml`。

相对冻结版YAML只修改`TRAIN.FREEZE_HEAD: true -> false`。原四源GOT10K_vottrain/LASOT/TNL2K/OTB99、1:1:1:0.2、每轮30000采样单元、batch16、VL_RATIO0.44、GROUNDING_RATIO0.11、验证数据和规模均不变。

Head LR1e-6，模块LR1e-4；AdamW weight decay1e-4；EPOCH=T_MAX=30；每轮验证保存。新模块仍只对flag=2启用，但其余样本不再是整体无训练价值：它们参与Head原始任务训练。

梯度裁剪按Head/模块**分别**执行，均用配置中的0.1，避免Head较大范数把模块梯度一起缩小。这与通常对全模型一次global clip不同，属于本版明确的优化策略，不隐瞒为“所有训练细节完全相同”。

验证分别记录Head原始IoU/损失，以及VL批次的中心IoU gain、有益/有害比例和残差。BBOX/NL验证中的新模块损失为0，但Head损失和IoU仍有意义。验证不是正式全序列AUC，也不自动按测试AUC选权重。

## 保存、恢复和测试必须配套

联合权重每轮保存**训练后Head全state_dict（含buffers）+细化模块+两个optimizer组+scheduler+随机状态**。冻结Backbone不重复保存，测试从明确的官方Epoch300加载它，再严格加载保存的Head和细化模块。

禁止把联合模块配官方Head直接测试。冻结版和联合版的variant/config/文件名不同；轮次、数据集、run_id、源权重和代码指纹检查均独立。工程合成权重不能用于正式训练恢复或正式测试。

文件名为`CenterJoint_ep0001.pth.tar`至`CenterJoint_ep0030.pth.tar`；修复后的目录为`change/center_joint_training_exact_metadata_20260930/<独立配置名>/`。联合版比冻结模块版大：Head约62.9MB，含AdamW两个moment后每轮预计约0.19GB，30轮约5.7GB，取决于实际获得梯度的参数数量。不是冻结模块版此前估计的8MB总量。

## 本轮检查结果（未训练）

- 两个真实VL活跃batch：Head梯度范数0.8759、1.3655；模块0.2015、0.2949，均有限且非零；新损失→Head及原始损失→新模块的梯度均为None。
- 强制纯BBOX的真实图像检查batch：Head有梯度，新模块无梯度。
- 全部5个验证loader读取成功，VL活跃样本32；原始Head损失和模块损失均有限。
- Backbone参数/buffers、Head BN统计，以及未执行更新的Head权重保持不变；优化器更新0步。
- 用明确标记的合成Head扰动检查保存/恢复，不是训练得到的权重。恢复Head和模块以及模拟Adam状态设备通过。
- 46帧零残差：与同一合成Head的native框、内部状态和prompt完全一致；46帧非零残差：只改变输出，内部状态和prompt不变。
- 单卡8个worker、两条训练源截短序列，共48帧，官方runner/writer与五指标函数通过。不是TNL2K700正式测试，也不是涨点证据。

最终真实检查报告：`change/rqc_dev_v3/center_joint_real_check_20260930_final.json`。
I/O工程检查：`change/rqc_dev_v3/center_joint_io_check_20260930.log`及时间戳目录内`check_report.json`。

## 确认后运行（本轮不执行）

服务器隔离项目仍是`/root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated`；环境仍是`uvltrack`。共享接入文件没有覆盖。

```bash
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uvltrack
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p terminal_logs
log="terminal_logs/train_center_joint_30ep_$(date +%Y%m%d_%H%M%S).log"
test ! -e "$log" || exit 1
nohup /root/miniconda3/envs/uvltrack/bin/python change/rqc_dev_v3/train_center_joint_20260930.py --train > "$log" 2>&1 < /dev/null &
echo "PID=$! 日志=$log"
```

首次启动输出目录已存在则拒绝覆盖。断点恢复显式加`--resume <该目录最后完整保存的CenterJoint_epNNNN.pth.tar>`，只支持轮次边界恢复；不删除partial、不截断日志或覆盖较新的权重。

训练完成后先固定Epoch30，示例新run_id93130：

```bash
config="baseline_base_head_output_center_safe_joint_bs16_30ep_20260930"
python change/rqc_dev_v3/test_center_joint_20260930.py \
  --dataset tnl2k --checkpoint "change/center_joint_training_exact_metadata_20260930/${config}/CenterJoint_ep0030.pth.tar" \
  --epoch 30 --runid 93130 --threads 8
python change/rqc_dev_v3/evaluate_center_joint_20260930.py \
  --dataset tnl2k --epoch 30 --runid 93130
```

测试完整性须覆盖700/700；中断的同一运行可显式`--resume-results`，仅相同manifest/checkpoint允许继续。报告缓存独立放在`change/center_joint_evaluation_20260930`，不能覆盖旧结果。

## 判断标准和已知风险

当前有开发集AUC正向信号，但LaSOT Precision下降仍需正式检验；不能宣称Head解冻已经修复它。中心限幅也无法修复已经选到远处干扰物的身份错误。新模块与Head同时训练可能改善综合定位，也可能因输入分布变化产生新的损失，这不是涨点保证。

本轮只实现和检查，不训练、不启动正式测试。确认后跑一次固定30轮；前期观察数值/梯度/有害修正，正常则继续，不再为了小样本过关反复更换结构。最终先看FULL综合结果，不自动启动完整OFF对照。
