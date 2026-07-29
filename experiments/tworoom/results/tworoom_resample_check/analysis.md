# TwoRoom resample cache stability check

## 实验目的

验证自然 region drift 是否依赖某一次 20k 样本抽样。这里重新抽取并编码了两份 TwoRoom latent cache，然后每份 cache 跑 5 个 analysis seeds。

## 实验设置

- 数据集：TwoRoom。
- 样本规模：每个 cache 20k。
- 新增 cache：sample_seed 1 和 sample_seed 2。
- 每个 cache 的 analysis seeds：0 到 4。
- GPU：CUDA_VISIBLE_DEVICES=6。
- batch size：1024。
- checkpoint：`/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt`。
- 输出目录：`experiments/real_gauge_drift/results/tworoom_resample_check`。

## 关键发现

两次重采样都复现了自然 region 的高 drift 模式。

doorway_corridor 在两次重采样中仍然是最干净的证据：

- sample_seed 1：PCA drift 0.8404 +/- 0.1450，PCA residual / IID 0.30 +/- 0.03，frame drift 1.2960 +/- 0.0016，frame residual / IID 1.02 +/- 0.00。
- sample_seed 2：PCA drift 0.6679 +/- 0.6828，PCA residual / IID 0.31 +/- 0.06，frame drift 1.2929 +/- 0.0019，frame residual / IID 1.03 +/- 0.00。

right_room 也稳定复现高 drift：

- sample_seed 1：PCA drift 1.4142 +/- 0.0000，PCA residual / IID 1.12 +/- 0.06，frame drift 1.4061 +/- 0.0057，frame residual / IID 1.44 +/- 0.00。
- sample_seed 2：PCA drift 1.3887 +/- 0.2977，PCA residual / IID 1.28 +/- 0.02，frame drift 1.4122 +/- 0.0059，frame residual / IID 1.44 +/- 0.00。

near_wall 也有高 drift，但 residual 明显升高，所以它更像 region shift / boundary distortion 混杂，不适合作为最干净的 gauge drift 证据。

## 需要小心的点

sample_seed 2 的 IID/common PCA drift 有一个 outlier。具体是 analysis_seed 4 的 IID PCA drift 达到 1.4142，而其他 analysis seeds 在 0.0022 到 0.0852 之间。这个现象说明 PCA-based 2D alignment drift 对 reference split 的符号或基方向翻转比较敏感。

因此，当前最稳的解读不是单看 PCA drift 的倍数，而是同时看：

1. IID non-overlap 的 typical drift 是否低。
2. natural region 的 frame drift 是否稳定高。
3. residual 是否没有同步大幅升高。

按这个标准，doorway_corridor 仍然是最强证据；right_room 是强但带 frame residual 混杂的证据；near_wall 不作为主证据。

## 当前结论

重新采样两次之后，TwoRoom 上的自然 region-dependent alignment drift 仍然存在。尤其 doorway_corridor 的模式跨样本稳定：drift 高，但 residual 与 IID 接近或更低。这支持我们继续进入下一步：检查这些 drift region 是否对应 predictor one-step error 和 multi-step rollout error 的升高。

## 下一步 predictor error 设计

下一步不要先改 predictor。先拿已有 pretrained encoder 和 world model predictor 做诊断。

具体做法：

1. 从 TwoRoom trajectory 中抽取连续片段，保留 observation、action、下一帧 observation、privileged state 和 region label。
2. 用 encoder 得到当前 latent 和下一步 latent。
3. 用 LeWM 原 predictor 在 latent 上做 one-step prediction。
4. 按 region 分组报告 one-step latent MSE，并和 region drift 做相关性。
5. 做 multi-step open-loop rollout：从同一个初始 latent 出发，喂真实 action 序列 rollout 5 到 20 步，比较预测 latent 与 encoder target latent 的误差增长。
6. 分别在 common/IID、doorway_corridor、right_room、near_wall 上报告 rollout error 曲线。
7. 成功证据不是“所有 OOD 都差”，而是 drift 高且 residual 不坏的 region 出现更明显 transition mismatch；doorway_corridor 是优先测试对象。

如果这个诊断成立，再进入同架构 predictor 对比：固定 encoder，训练 baseline predictor 和你的 soft gauge-equivariant predictor，在相同 train data、相同优化配置下比较 one-step 与 multi-step rollout。
