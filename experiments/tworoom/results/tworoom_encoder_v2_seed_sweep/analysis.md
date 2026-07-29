# TwoRoom cache analysis seed sweep 结果分析

## 实验设置

- 数据集：TwoRoom。
- latent 来源：已缓存的 20k encoder latent。
- 分析种子：10 个，seed 0 到 9。
- 每个 seed 改变 reference/common 与 IID non-overlap 的分析切分。
- 没有重新训练 encoder，也没有重新编码图像；这次检查的是分析切分稳定性，不是完整数据采样稳定性。
- 关键对照：IID non-overlap。它用于估计“同分布、无自然 region shift 时，Procrustes / subspace alignment 自身会产生多大 drift”。

## 核心结果

| split | PCA drift | PCA residual / IID | frame drift | frame residual / IID | latent-to-state R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iid_nonoverlap | 0.0326 +/- 0.0271 | 1.00 +/- 0.00 | 0.1356 +/- 0.0086 | 1.00 +/- 0.00 | 0.9999 +/- 0.0000 |
| doorway_corridor | 0.7725 +/- 0.2325 | 0.24 +/- 0.03 | 1.2936 +/- 0.0030 | 1.03 +/- 0.00 | 1.0000 +/- 0.0000 |
| right_room | 1.3416 +/- 0.2038 | 1.16 +/- 0.02 | 1.4161 +/- 0.0025 | 1.45 +/- 0.00 | 0.9984 +/- 0.0006 |
| left_room | 0.3630 +/- 0.2142 | 1.15 +/- 0.03 | 1.4201 +/- 0.0020 | 1.47 +/- 0.00 | 0.9993 +/- 0.0001 |
| goal_other_side | 0.7227 +/- 0.1658 | 1.17 +/- 0.03 | 1.4125 +/- 0.0024 | 1.43 +/- 0.00 | 0.9922 +/- 0.0002 |
| near_wall | 1.1909 +/- 0.2376 | 1.85 +/- 0.05 | 1.4967 +/- 0.0016 | 1.47 +/- 0.00 | 0.9986 +/- 0.0002 |

## 按实验设计解读

这组结果支持“自然 region 会诱导 encoder state-aligned subspace / frame alignment drift”，但还不能直接说完整 latent 空间里的全局正交矩阵变了。原因是 TwoRoom 的 privileged state 维度很低，而 encoder latent 维度高，所以当前测的是 state-aligned 子空间上的 drift。

最强证据是 doorway_corridor。它的 PCA drift 明显高于 IID non-overlap，但 PCA residual 反而低于 IID，frame residual 也几乎等于 IID。这符合我们预先设定的成功模式：不是 encoder 在该 region 简单学坏了，而是同样可对齐的状态信息在该局部 region 中对应了不同的 alignment。

right_room 是第二强证据。它的 PCA drift 非常高，PCA residual 只比 IID 高约 16%。这说明右房间很可能存在稳定的 region-dependent alignment 变化。不过它的 frame residual 明显高于 IID，所以在高维 frame 层面仍有一定 region shift / representation distortion 混杂。

goal_other_side 和 left_room 是中等强度证据。它们的 drift 明显高于 IID，但 residual 也升高，说明这里既可能有 alignment drift，也可能混入了自然 OOD 区域带来的非线性表示变化。

near_wall 不适合作为最干净主证据。它 drift 很高，但 residual 也明显升高，说明靠墙区域可能是 representation distortion、边界效应或数据覆盖差造成的混合现象。它可以作为 failure-prone region 或误差相关性分析对象，但不应该作为最强 gauge drift 证据。

latent-to-state R2 在所有 split 上都接近 1，说明 privileged state 信息可以从 encoder latent 中线性读出。这对当前分析是有利的：我们不是在一个完全无法从 latent 恢复 state 的坏表示上硬做 alignment。

## 当前结论

在 TwoRoom 的自然区域切分上，IID non-overlap drift 很低，而 doorway、right room、goal-side 等自然 region 的 drift 明显更高。尤其 doorway_corridor 同时满足“drift 高、residual 不高”的成功标准，是目前最干净的证据。

更准确的 claim 应该是：TwoRoom pretrained encoder 的 state-aligned latent subspace 在自然状态区域之间存在 dataset/region-dependent alignment drift。这个结果支持我们继续研究 predictor 是否会在这些 drift region 上出现 transition mismatch。

## 下一步

1. 对 TwoRoom 重新抽样生成多个 latent cache，检查数据采样层面的稳定性。
2. 在 doorway_corridor 与 right_room 上测 predictor one-step error 和 multi-step rollout error，检查 drift 是否预测 transition mismatch。
3. 对 PushT 做同样的 region drift 发现实验，重点看 contact / non-contact、object angle bins 和 near-goal / far-goal。
4. 如果要引入 soft equivariant predictor，优先比较同一个 predictor 架构下的多步 rollout，而不是只看 equivariance loss 指标。
