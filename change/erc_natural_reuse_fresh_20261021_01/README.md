# FreshNaturalERC：全新自然框60轮

用户已明确确认：冻结RQC30，ERC随机初始化，第1轮完整训练60轮；不加载旧ERC，不自动测试。

本目录与E14续训目录隔离。34987参数；复用RQC现有特征和hidden context，不增加Backbone前向；从E1起仅自然RQC框，不含五锚点扩增。

固定原TRAIN成员、30000 tickets/epoch、batch16、VL .44、每VL ticket两观测、warmup16/clip192、seed20261016、AdamW4e-5/wd1e-4/clip1、fresh cosine60。优化器/scheduler/RNG不从旧实验恢复。

工程检查不是收益证据；自然因果轨迹收集与原RQC的GT-jitter批训练不同，不能承诺同等速度或正收益。训练中统计不等于官方AUC。固定E60，不挑best、不自动重训或测试。
