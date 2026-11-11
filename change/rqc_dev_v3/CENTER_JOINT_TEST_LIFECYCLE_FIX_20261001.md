# Epoch30 TNL2K FULL：显存生命周期修复与续测

## 原测试状态

配置`baseline_base_head_output_center_safe_joint_bs16_30ep_20260930`，权重来自`change/center_joint_training_exact_metadata_20260930`的`CenterJoint_ep0030.pth.tar`。

run_id=100130，NLBBOX，8个测试进程。2026-10-01 09:39:49开始，10:27:20因缺失序列完整性检查失败退出。此前多次CUDA OOM被官方runner捕获，故不能把其打印的Done当作700条全部成功。完整结果397/700，评估未启动。

官方预测框保存和指标公式没有修改。已保存的397条结果无需重跑，错误序列未生成完整结果。

## 本次只修改资源生命周期

原close把保存的bound method重新写回network实例，形成network→method→network引用环。Python GC不按CUDA显存压力触发；8个进程各自缓存/峰值叠加，容易使同一GPU耗尽。

新增`center_joint_lifecycle_20261001.py`：

- 序列结束后删除实例级forward_test覆盖，恢复类方法descriptor，不再写入绑定方法形成自引用。
- 移除被动特征hook，清除已经完成跟踪的模型、模板、prompt及响应引用。
- 序列边界执行gc.collect/empty_cache，异常路径同样清理；不在帧内改变预测计算。

新增`test_center_joint_lifecycle_safe_20261001.py`只给原测试入口替换生命周期适配器。权重、YAML、Head、中心模块、原响应选点、返回框、内部状态、结果writer和指标函数不变。训练时记录的原源码指纹也没有篡改或绕过；新adapter另存hash到续测sidecar。

## 回归证据与限制

诊断中人为暂停Python GC以隔离引用释放：旧实现三次close后仍保留3个网络，CUDA分配分别增长647.66/1295.32/1942.98 MiB；新实现不保留网络，三次增长均为0。此受控诊断证明引用回收风险，不表示正式测试中GC永久关闭，也不能据此认定OOM只有一个原因。

使用正式E30权重在两条真实TNL2K序列前24帧对比旧/新实现；要求46个后续帧的输出框、内部状态和prompt完全一致。再用5次真实短序列核对结束后的CUDA allocations不累积。报告为`center_joint_lifecycle_check_20261001.json`；全部通过后才允许续测。

## 续测保护

同一run_id100130、同一checkpoint SHA256 `d95900188a2ed05885fc1d81325448b9f3710944613dd25ec08bdd1dd6cf4156`。先核对现有manifest完全一致、已存在预测完整且有限，记录其bbox/time文件hash与mtime。

4进程、单卡，显式`--resume-results`；官方runner跳过已完成序列，只补缺失结果。结束后再次核对原397条文件hash与mtime未变，700/700完整后才用原评估入口计算五指标。

独立启动脚本：`resume_center_joint_e30_tnl2k_full_20261001.sh`。

新日志：

- `terminal_logs/launch_center_joint_e30_tnl2k_FULL_run100130_resume4_20261001.log`
- `terminal_logs/test_center_joint_ep30_tnl2k_FULL_run100130_resume4_20261001.log`
- `terminal_logs/eval_center_joint_ep30_tnl2k_FULL_run100130_resume4_20261001.log`

旧失败日志保留。不重训，不重新跑已完成序列，不创建定时监控。后续只在同一700条完整结果上评估，不能拿397条部分结果当最终AUC。

## 实际检查与恢复记录

2026-10-01检查已通过：46帧输出框、内部状态和prompt逐项完全相同；5次序列释放后的CUDA allocated均为8.18 MiB，没有累积。检查没有执行优化器步骤。

10:56:53（北京时间）已开始后台续测，启动器PID6437，测试主进程PID6450，4个工作进程。续测前确认397条完整、303条缺失；使用同一正式Epoch30权重、run_id100130和原结果目录。10:57:21采样GPU利用率77%，显存5210/24564 MiB。此处只是启动后的采样，不代表全程峰值或已完成测试。
