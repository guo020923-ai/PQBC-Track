# 中心安全型输出细化：30轮受控训练准备

## 当前结论

正式入口已准备，工程检查通过；本轮没有启动正式训练。默认入口只检查，必须显式添加 `--train` 才更新参数。

“可启动”只表示实现、数据接入、梯度和运行入口满足条件，不表示已经证明正式测试涨点。此前开发结果仍有LaSOT Precision下降，不能声称问题已全部消失。

## 为什么是这次改动

旧候选排序在BQ/JQ正式TNL2K测试中没有获得正收益；JQ的AUC为62.15，低于官方本地复现62.73。继续让Head漂移、反复更换响应融合式，不能直接解决这个事实。

局部框细化实验提供了更具体的证据：把细化框写回状态会放大轨迹误差；只改变输出则改善。尺寸修正的开发收益为负，中心修正为正。因此保留官方响应选点和框尺寸，只学习小幅中心修正，并不把修正结果反馈进内部搜索状态。

当前模块是**输出定位细化**，不是原先的“仅候选重排序、不移动框”。旧ERC及旧RQC均关闭，不应继续沿用原先的两个模块定义来描述它。它也不预测候选IoU，不应包装为已经实现的IoU质量排序头。

## 固定方法

- 官方Epoch300权重作为唯一基础，Backbone和Head参数冻结并保持eval，包含running buffers。
- 官方C/T×Hann选点不变。在选中的位置提取原Head的288维定位特征。
- LayerNorm→64维MLP→2维中心残差，共19,202参数；末层零初始化，中心偏移限幅原框宽高的10%。
- 原crop框宽高不变；原图边界clip仍可能改变输出尺寸。
- 训练：2GIoU +5原图归一化xyxy L1 +相对原框IoU损害项 +中心误差损害项。两项损害权重均为1。新L1坐标定义不是官方原crop L1，不混称同一损失。
- 特征、原框、GT隔离梯度；测试没有GT输入或数据集专用规则。
- 只修正返回框；native state、搜索裁剪和prompt更新使用官方原预测。不把它宣传为已改善内部闭环轨迹。
- 官方训练标注采用`(size-1)/2`，测试映射采用`size/2`。入口分别记录标注逆映射和测试解码原点；新框监督按实际测试解码计算，避免半个搜索像素的系统偏移。原图、标注增强及随机数流程不变。

## 数据和训练预算

独立YAML：`experiments/uvltrack/baseline_base_frozen_head_output_center_safe_bs16_30ep_20260930.yaml`。

DATA与原JQ配置逐项严格比较，相同：GOT10K_vottrain、LASOT、TNL2K、OTB99，比例1:1:1:0.2，每轮30,000个采样单元、每单元2张search、batch16；VL_RATIO=0.44、GROUNDING_RATIO=0.11，其他采样/增强不变。30,000不是30,000个VL单元：新模块只监督flag=2，期望约13,200个VL单元/轮，具体数量随机波动。

保留原验证数据配置，VALTRACK/VALVL各6400，间隔1轮。旧tracking/grounding验证仍读取，但新模块对flag=0/1不启用，其新损失为0；只有`vl_*`验证提供新模块的有效IoU/损害统计。它是采样验证，不是全序列测试AUC，也不会自动挑选“最佳AUC权重”。

30轮与T_MAX=30固定，AdamW，模块LR=1e-4，weight_decay=1e-4，clip_norm=0.1，单卡、不启用DDP。YAML里保留的Head/RQC旧LR字段不进入优化器。此前缓存原型的weight_decay=0、20轮×50batch×64，并非同一个训练预算，不能拿缓存训练当正式30轮。

## 已通过的检查

- 30组增强：图像、原标注及Python/NumPy/Torch随机状态与native一致，覆盖水平翻转和padding；新框解码直接与官方`map_box_back`及`clip_box`比较通过。
- 官方来源缓存100条、3861样本：源权重SHA256一致、零残差恢复、有限非零梯度、上游梯度隔离。
- 真实训练batch反向与全部5个验证loader前向；优化器不更新，冻结buffers不变，只有19,202个新参数可训练。
- 48帧零残差与官方输出逐坐标完全一致；另外48帧合成非零残差只改变返回框，native state和prompt仍完全一致。
- 单卡8个worker运行两条训练源截短序列共48帧，官方runner/writer与五指标评估函数通过。这里只是工程冒烟，不是官方测试结果。
- 保存/恢复控制流程检查通过：配置/代码/源权重不匹配拒绝恢复，新轮次存在时拒绝覆盖；合成权重不允许进入正式测试；不完整预测文件不允许算作完成。

具体检查报告保存在隔离目录`change/rqc_dev_v3/center_formal_ready_check_20260930_v3.json`及同名log。检查中的所有优化器更新均为0。

## 尚未证实的效果

此前JQ来源的20条训练源开发序列，CenterSafe在官方指标定义下：平均AUC +0.193个百分点，LaSOT +0.132、TNL2K +0.254；LaSOT Precision -1.014，TNL2K +1.003。它们不是官方测试结果，也不是相对官方Epoch300的已确认收益。旧40条确认集已用于反馈，不再作为新结构的独立最终证明。

这给出一次固定结构30轮实验的理由，不给出“必涨点”的结论。正式结果应包含整数预测文件上的AUC/P/NormP/OP50/OP75；官方writer仍保留原来的整数保存规则，不能直接把浮点开发分数当正式结果。

## 服务器位置与入口

隔离项目：`/root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated`。

源权重：`/root/autodl-tmp/UVLTrack-master/checkpoints/train/uvltrack/baseline_base/UVLTrack_ep0300.pth.tar`。

源SHA256：`146599daa765c3f14f67d433a5c5b19501f7d0af583fb6e09ebefd123757e6c2`。

独立入口位于`change/rqc_dev_v3`：train_center_formal、test_center_formal、evaluate_center_formal，日期均为20260930。没有覆盖共享registry、actor、tracker、原YAML或历史结果。这不是直接运行官方`tracking/train.py`；它复用官方模型和原四源loader，使用新的独立actor/训练循环。测试复用官方dataset/runner/writer；评估直接调用官方指标函数，只把报告缓存重定向到新目录。

## 确认后启动训练（本轮未执行）

```bash
cd /root/autodl-tmp/UVLTrack-master/change/rqc_jq_20260929_isolated
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uvltrack
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE OPENCV_FOR_THREADS_NUM=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p terminal_logs
log="terminal_logs/train_center_safe_30ep_$(date +%Y%m%d_%H%M%S).log"
test ! -e "$log" || exit 1
nohup /root/miniconda3/envs/uvltrack/bin/python change/rqc_dev_v3/train_center_formal_20260930.py --train > "$log" 2>&1 < /dev/null &
echo "PID=$! 日志=$log"
```

默认权重目录：`change/center_safe_training_20260930/baseline_base_frozen_head_output_center_safe_bs16_30ep_20260930/`。新训发现该目录已存在就拒绝启动，不静默重训覆盖。

每轮保存`CenterSafe_ep0001.pth.tar`至`CenterSafe_ep0030.pth.tar`，仅保存新模块和恢复状态。合成空优化器检查文件约94KB；正式AdamW两组moment预计单轮约0.25MB，30轮约8MB量级，不再重复保存完整大baseline。此估计不包含缓存、日志和预测。

恢复使用同一训练入口，添加`--train --resume <该目录中最后完整保存的CenterSafe_epNNNN.pth.tar>`；从下一轮开始，恢复optimizer、scheduler及随机状态。只支持轮次边界恢复；不承诺GPU数值逐位复现。若发现更新的权重、partial文件或日志轮次不对齐，拒绝续训并保留现场，不自动删除/截断文件。

## 训练后测试与评估（本轮未执行）

仍在上述隔离项目和uvltrack环境下，先检查Epoch30确实已保存。示例使用新run_id 93030，入口会拒绝覆盖已有运行目录：

```bash
config="baseline_base_frozen_head_output_center_safe_bs16_30ep_20260930"
checkpoint="change/center_safe_training_20260930/${config}/CenterSafe_ep0030.pth.tar"
python change/rqc_dev_v3/test_center_formal_20260930.py \
  --dataset tnl2k --checkpoint "$checkpoint" --epoch 30 --runid 93030 --threads 8
python change/rqc_dev_v3/evaluate_center_formal_20260930.py \
  --dataset tnl2k --epoch 30 --runid 93030
```

中断的同一测试可显式加`--resume-results`，仅当checkpoint指纹和manifest完全相同才继续；保留有效的逐序列文件。正式测试末尾必须出现`TEST_COMPLETE`且覆盖700/700，不以原runner的`Done`作为唯一完成依据。

评估缓存/报告放入独立`change/center_safe_evaluation_20260930/<config>_tnl2k_e0030_run93030/`；已有报告会拒绝覆盖。没有修改AUC阈值、可见性处理或Precision定义。当前先固定末轮实验，不用官方测试AUC反复选轮次。

## 下一步边界

收到训练确认后，按固定结构启动30轮，检查早期NaN/梯度/模块有效样本及验证损害趋势；正常则继续完成，不再反复换结构跑3轮。若存在数值故障或持续明显有害动作，则明确暂停诊断，而不是为了满足“可训练”忽略风险。先完成这次正式实验，再决定是否值得后续扩充轮次或增加第二模块。
