# RQC-style epoch-supervised ERC v1

This run reuses the original RQC training population and its 30-epoch training
discipline, but it does not copy the RQC center-refinement module or objective.

- Frozen source: verified RQC Epoch 30.
- Active samples: the original loader's vision-language TRAIN population only;
  no benchmark test split is read for training or model selection.
- One fixed cache contains 30,000 causal RQC30 observations. The expensive
  frozen tracker is executed once per cached frame, not twice per optimizer
  step and not again in later epochs.
- The ERC model is stateless and scores all 25 executable actions. KEEP is
  anchored to zero and inference changes the RQC box only for a strictly
  positive predicted relative score.
- Labels are official-style success-AUC21 utility with a 1e-3 continuous-IoU
  tie breaker. Labels are constructed after the frozen forward and never enter
  the model input.
- Training is 30 epochs, 30,000 examples/epoch, batch 16, 1,875 steps/epoch.
  Epoch 30 is reported; no best-checkpoint selection is allowed.
- Progress is atomically persisted every 25 optimizer steps with elapsed time,
  throughput, and ETA. Every epoch has a resumable checkpoint.

The cache deliberately fixes the 30,000 sampled observations across epochs.
Therefore this is RQC-style supervised optimization, not an exact replay of
RQC's independently re-jittered image crops in every epoch. It learns immediate
per-frame action utility and does not claim the long-horizon credit assignment
of paired closed-loop reinforcement learning.
