# 联合中心修正训练：裁剪元数据精度修复

## 已确认的根因

第一次训练于2026-09-30 15:31北京时间退出，数据读取完成约1140个batch；最后按50步间隔输出的日志是Epoch1 step1100。没有完整轮次、验证记录或checkpoint，不能从这次失败中断点续训。

使用相同随机种子和原流程、零优化器更新，复现到第1141个batch触发相同断言。失败extract_box为`[0.756011962890625,455.1298522949219,125.260498046875,679.5283813476562]`。

- 官方sample_target：先tolist得到Python浮点数，再算宽高乘积`85118.06348459423`，最终crop side为1168。
- 新增旧元数据：先在float32 Tensor中算乘积`85118.0625`，最终crop side为1167。
- 差1个裁剪像素导致标注逆映射最大约0.498像素差异。不是训练NaN，不是模型梯度爆炸，也不是应当忽略的日志告警。

## 修改边界

新增`center_joint_metadata_exact_20260930.py`，只在联合版接入。先保留原有jitter/随机数流程，再严格按官方sample_target的Python浮点数计算顺序记录side和对应origin。

不修改官方采样/图像裁剪/GT/增强；不放宽`rtol=2e-4,atol=0.004`；不跳过失败样本。冻结版原metadata文件原样保留。联合actor和训练入口修改前保存在`center_joint_before_metadata_fix_20260930`。

YAML名称、内容、四个训练源、样本量、batch16、30轮学习率周期、Head1e-6、模块1e-4及损失均不变。源码身份记录增加新metadata文件hash。

## 检查与重启要求

1. 边界回归：1167/1168差异明确复现，修复后为1168；原始图像/标注和随机数状态逐项一致，覆盖水平翻转及非翻转。
2. 完整训练数据轮次及全部5个验证loader读取检查，仍保留每样本坐标一致性断言。检查无优化器更新。
3. 真实batch有限损失、梯度隔离、Backbone/BN固定检查重跑，必要的Head保存/加载及tracker等价性检查重跑。
4. 全部通过后才重新从官方Epoch300开始30轮。旧失败目录和旧日志保留，不伪称断点续训。

新默认输出目录：

`change/center_joint_training_exact_metadata_20260930/baseline_base_head_output_center_safe_joint_bs16_30ep_20260930/`

每轮权重仍为`CenterJoint_epNNNN.pth.tar`；源码/YAML/初始化校验通过后才能用于测试。

训练入口仍为`change/rqc_dev_v3/train_center_joint_20260930.py --train`，单卡uvltrack环境。后台日志用新时间戳名称，不覆盖旧日志。不自动启动测试，不创建定时监控。

正式测试权重路径须改为新输出目录，例如：

```bash
config="baseline_base_head_output_center_safe_joint_bs16_30ep_20260930"
python change/rqc_dev_v3/test_center_joint_20260930.py \
  --dataset tnl2k \
  --checkpoint "change/center_joint_training_exact_metadata_20260930/${config}/CenterJoint_ep0030.pth.tar" \
  --epoch 30 --runid 93130 --threads 8
python change/rqc_dev_v3/evaluate_center_joint_20260930.py \
  --dataset tnl2k --epoch 30 --runid 93130
```

这里只是工程数值修复，不能将其本身宣传为新模块创新或已获得AUC提升。

## 实际完成记录

- 失败复现：`center_metadata_diagnosis_20260930.json/log`，零优化器更新，在原先相同前缀的1140个成功batch后触发同一错误。
- 完整数据回归：`center_joint_metadata_full_epoch_20260930.json/log`，2740个batch、86628张搜索图；训练1875、tracking验证400、三个grounding验证3/44/18、VL验证400，全部完成。用时327秒，未训练。
- 有限损失和梯度检查：`center_joint_real_check_metadata_fixed_20260930.json/log`，通过。
- 保存/恢复和tracker接口：`center_joint_io_metadata_fixed_20260930.log`，46帧零残差原路径精确等价，46帧合成非零残差只改变输出；8进程两条截短训练源序列接口通过。不是正式AUC测试。
- 2026-09-30 15:53:24北京时间重新启动正式30轮，PID33766，日志`terminal_logs/train_center_joint_metadata_fixed_30ep_20260930_155324.log`。
- 未创建定时监控，不自动启动正式测试。
