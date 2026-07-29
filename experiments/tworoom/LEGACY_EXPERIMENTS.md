# 轨迹偏差实验

为了直观显示数据集上不同区域动力学规律不一致，我们采用 open-loop latent rollout 的方式比较不同 predictor 的轨迹偏差。

## 准备工作

1. 数据集划分：取 train global 作 reference。下方表格为 **legacy quantile** 全数据 region predictor；新实验见 `scripts/run_geometry_*.sh`。
2. 模型训练：在下载好的 LeJEPA 权重上，对每个 natural region 冻结 encoder、只 finetune predictor。训练脚本见 `trajectory.py`；统一配置为 `epochs=30, lr=5e-5, batch_size=128, weight_decay=1e-3, history_size=3`。

### 全局 baseline

| 名称 | 路径 |
|------|------|
| `P_train_global`（下载的全局 LeWM） | `/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt` |

### 区域 predictor 路径

| 名称 | 路径 | 训练样本数 | eval loss |
|------|------|-----------|-----------|
| `P_common` | `experiments/tworoom/results/tworoom_trajectory_predictors/P_common_object.ckpt` | 256579 | 0.00335 |
| `P_doorway_corridor` | `experiments/tworoom/results/tworoom_trajectory_predictors/P_doorway_corridor_object.ckpt` | 75988 | 0.00561 |
| `P_left_room` | `experiments/tworoom/results/tworoom_trajectory_predictors/P_left_room_object.ckpt` | 259983 | 0.00188 |
| `P_near_wall` | `experiments/tworoom/results/tworoom_trajectory_predictors/P_near_wall_object.ckpt` | 210150 | 0.00232 |
| `P_right_room` | `experiments/tworoom/results/tworoom_trajectory_predictors/P_right_room_object.ckpt` | 255132 | 0.00189 |

每个 region 还对应：

```bash
experiments/tworoom/results/tworoom_trajectory_predictors/P_{region}_embeddings.npz
experiments/tworoom/results/tworoom_trajectory_predictors/P_{region}_starts.npy
experiments/tworoom/results/tworoom_trajectory_predictors/P_{region}_object.json   # 含完整 epoch history
```

训练日志：

```bash
experiments/tworoom/results/tworoom_trajectory_predictors/common_train.log
experiments/tworoom/results/tworoom_trajectory_predictors/other_regions_train.log
experiments/tworoom/results/tworoom_trajectory_predictors/manifest.json
```

### 收敛曲线（train epoch loss）

下表为每个 predictor 在各自 region 数据上的 **batch 平均 train loss**；`eval loss` 为训练结束后全量 eval（无 shuffle）。

| region | epoch 1 | epoch 10 | epoch 20 | epoch 30 | eval loss | 末 5 epoch 降幅 | 收敛判断 |
|--------|---------|----------|----------|----------|-----------|----------------|----------|
| common | 0.0217 | 0.0142 | 0.0110 | 0.0093 | 0.00335 | 7.8% | 收敛 |
| doorway_corridor | 0.0304 | 0.0205 | 0.0167 | 0.0144 | 0.00561 | 7.4% | 基本收敛 |
| left_room | 0.0180 | 0.0110 | 0.0085 | 0.0071 | 0.00188 | 8.8% | 收敛 |
| near_wall | 0.0200 | 0.0127 | 0.0104 | 0.0090 | 0.00232 | 7.0% | 基本收敛 |
| right_room | 0.0177 | 0.0108 | 0.0085 | 0.0071 | 0.00189 | 8.1% | 收敛 |

**结论**：

1. 五个 region predictor 的 train loss 均从 epoch 1 到 30 **单调下降**，无发散或震荡。
2. 末 5 个 epoch 仍有约 **7–9%** 的缓慢下降，尚未完全 plateau，但已进入尾部；`doorway_corridor` / `near_wall` 末 epoch 有极小波动（<0.2%），整体可视为基本收敛。
3. `eval loss` 均显著低于最后一个 epoch 的 train loss，说明拟合稳定；`doorway_corridor` 样本最少、eval loss 相对最高，与其样本量一致。
4. 当前 **30 epoch 配置可用**；若需进一步压低 loss，可加长训练，但收益可能有限。

## 实验设定

在 **held-out test split** 上，用不同 predictor 以测试集 transition 为起点做 **10 步 open-loop rollout**：

- **样本**：与 Stage 1 相同的 90/10 train/test 划分（`split_seed=3072`），使用 **全部 held-out test pool（约 32517 条 transition）**；**不按 natural region 过滤**。
- **初始 history**：前 3 步 latent 用 frozen encoder 的 ground truth（`history_size=3`，与训练配置一致）。
- **动作**：每步使用数据集中的 ground-truth action（teacher forcing），不引入 planner / policy。
- **GT**：frozen encoder 编码出的真值 latent，用于计算 `vs_gt` 误差。
- **指标**：
  - `pairwise_mse`：同一步上两个 predictor 预测 latent 的 MSE，再对样本平均；
  - `A_vs_gt` / `B_vs_gt`：各自预测相对 encoder GT 的 MSE。

脚本：

```bash
experiments/tworoom/trajectory_deviation.py
```

运行示例（`P_common` vs 训练集全局 predictor）：

```bash
python experiments/tworoom/trajectory_deviation.py \
  --predictor-a experiments/tworoom/results/tworoom_trajectory_predictors/P_common_object.ckpt \
  --predictor-b /data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt \
  --predictor-a-name P_common \
  --predictor-b-name P_train_global \
  --max-steps 10 \
  --test-max-samples 0 \
  --save-test-cache experiments/tworoom/cache/tworoom_trajectory_test_full_transitions.npz \
  --out-dir experiments/tworoom/results/tworoom_trajectory_deviation_common_vs_train \
  --device cuda
```

## 已完成结果：各 region predictor vs `P_train_global`

在 **完整 test pool（32517 条）** 上，将五个全数据集 region predictor 分别与下载的全局 predictor 对比。

测试集 latent 缓存：

```bash
experiments/tworoom/cache/tworoom_trajectory_test_full_transitions.npz
```

### 结果文件路径

| 对比 | CSV | JSON | 日志 |
|------|-----|------|------|
| `P_common` vs `P_train_global` | `experiments/tworoom/results/tworoom_trajectory_deviation_common_vs_train/trajectory_deviation.csv` | `.../trajectory_deviation.json` | `.../run_full_test.log` |
| `P_doorway_corridor` vs `P_train_global` | `experiments/tworoom/results/tworoom_trajectory_deviation_doorway_corridor_vs_train/trajectory_deviation.csv` | `.../trajectory_deviation.json` | — |
| `P_left_room` vs `P_train_global` | `experiments/tworoom/results/tworoom_trajectory_deviation_left_room_vs_train/trajectory_deviation.csv` | `.../trajectory_deviation.json` | — |
| `P_near_wall` vs `P_train_global` | `experiments/tworoom/results/tworoom_trajectory_deviation_near_wall_vs_train/trajectory_deviation.csv` | `.../trajectory_deviation.json` | — |
| `P_right_room` vs `P_train_global` | `experiments/tworoom/results/tworoom_trajectory_deviation_right_room_vs_train/trajectory_deviation.csv` | `.../trajectory_deviation.json` | — |

512 条 smoke 结果（仅供参考）：`experiments/tworoom/results/tworoom_trajectory_deviation_common_vs_train_smoke/`

### 总表：各 region predictor 对 `P_train_global` 的 pairwise 偏差（MSE）

在完整 test pool（32517 条）上，每步为区域 predictor 与 `P_train_global` rollout 预测的 latent MSE（对样本平均）。

| step | `P_common` | `P_doorway_corridor` | `P_left_room` | `P_near_wall` | `P_right_room` |
|------|------------|----------------------|---------------|---------------|----------------|
| 1 | 0.105 | 0.116 | 0.276 | 0.072 | 0.283 |
| 2 | 0.185 | 0.227 | 0.503 | 0.108 | 0.485 |
| 3 | 0.273 | 0.384 | 0.635 | 0.151 | 0.635 |
| 4 | 0.364 | 0.559 | 0.713 | 0.205 | 0.732 |
| 5 | 0.458 | 0.712 | 0.776 | 0.268 | 0.802 |
| 6 | 0.551 | 0.828 | 0.835 | 0.339 | 0.866 |
| 7 | 0.638 | 0.914 | 0.896 | 0.412 | 0.930 |
| 8 | 0.717 | 0.981 | 0.958 | 0.483 | 0.995 |
| 9 | 0.787 | 1.036 | 1.020 | 0.548 | 1.059 |
| 10 | 0.846 | 1.083 | 1.083 | 0.605 | 1.124 |

### 初步解读

1. 五个 region predictor 相对 `P_train_global` 的 **pairwise 偏差均随 rollout 步数单调上升**，说明区域特化动力学与全局动力学不同，且 open-loop 下分叉随步数累积。
2. **`P_near_wall`** 全程偏差最小（step 10 = 0.605）；**`P_left_room` / `P_right_room`** 从 step 1 起偏差就明显更大（0.28 vs 0.11 左右）。
3. 当前结果是 **全 test pool 平均**，不是 `test ∩ region`；若要验证区域特化效应，下一步应对 test 样本按 natural region 分层报告。

# multipredictor jepa

## 准备工作（geometry 划分）

1. **Natural region 划分**：使用 `--region-split-mode geometry`。边界为固定任务几何（图像 224×224、playable 边界 14、墙心 x=112、墙宽 10），**不依赖数据分位数，无 test 泄露问题**。具体标准（agent 位置为 `(x, y)`）：
   - `left_room`：`x < 107`；
   - `doorway_corridor`：`107 ≤ x ≤ 117`；
   - `right_room`：`x > 117`；
   - `near_wall`：距 playable 外边界不超过 15px，即 `x ≤ 29` 或 `x ≥ 194` 或 `y ≤ 29` 或 `y ≥ 194`；
   - `common`：房间内部相对 playable 边界和中墙各内缩 20px，即 `34 ≤ y ≤ 189`，且满足 `34 ≤ x ≤ 87`（左侧内部）或 `137 ≤ x ≤ 189`（右侧内部）；
   - `goal_other_side`：agent 与 target 分居墙心 `x=112` 两侧，即 `(x_agent ≤ 112 < x_target)` 或 `(x_target ≤ 112 < x_agent)`。

   这些集合不是全部互斥：`left_room` / `right_room` 会分别包含对应侧的 `common` 和 `near_wall`。训练时每个 predictor 独立使用其 region mask；priority5 推理时按 `doorway_corridor > near_wall > common > right_room/left_room` 消解重叠。
2. **模型训练**：在下载好的 LeJEPA 权重上，对每个 **geometry region ∩ train split** 冻结 encoder、只 finetune predictor。统一配置：`epochs=30, lr=5e-5, batch_size=128, weight_decay=1e-3, history_size=3`。输出前缀为 `train_`（如 `P_train_common_object.ckpt`）。

   **训练可复现协议（2026-07-13 起强制）**

   | 项 | 值 | 作用 |
   |----|-----|------|
   | `split_seed` | **3072** | train 90% 划分（`train_global_reference_starts.npy`） |
   | `seed` | **42** | predictor FT：`DataLoader` shuffle + `torch`/`numpy`/`random` RNG |
   | cuDNN | `deterministic=True`, `benchmark=False` | CUDA 算子确定性 |
   | 环境 | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 部分 BLAS 算子确定性 |

   实现：`trajectory.py` 的 `set_training_seed()`（须在 `load_encoder()` 之前调用）+ `train_region_predictor()` 内 `Generator.manual_seed(seed)`。所有 geometry / cluster predictor 脚本须传 `--seed 42`。**重训 predictor 时复用已有 `P_train_*_embeddings.npz`（encoder 冻结，与 FT seed 无关）；勿加 `--force-reencode`。**

   **并行训练注意（2026-07-13）**

   - 多 region **不能**共写同一 `OUT_DIR`（会竞争 `manifest.json` / `train_global_reference_starts.npy` / 阈值文件）。
   - 使用 `run_geometry_train_one_region.sh`：每 job 写入 `MAIN_DIR/_work/${REGION}_${EPOCHS}ep_seed${SEED}/`，结束后仅把 `P_train_{region}*` checkpoint **汇总**到 `MAIN_DIR`。
   - latent kmeanspp 并行须显式传 GPU：`GPU=1 OUTER_SEED=1 bash scripts/run_latent_kmeanspp_train_predictors_50ep.sh`（内层脚本 `GPU` 为必填）。
   - 中断/半成品用 `scripts/archive_aborted_training.sh` 归档为 `*_aborted_*`，**勿与正式结果混用**；embedding 缓存在 `tworoom_geometry_train_region_predictors/` 保留复用。

   单 job 示例：

   ```bash
   GPU=0 REGION=common EPOCHS=30 bash experiments/tworoom/scripts/run_geometry_train_one_region.sh
   GPU=1 REGION=left_room EPOCHS=50 SELECT_BEST=1 SAVE_EPOCHS=20,30,40,50 bash experiments/tworoom/scripts/run_geometry_train_one_region.sh
   GPU=0 OUTER_SEED=0 bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh
   GPU=1 OUTER_SEED=1 bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh
   GPU=2 OUTER_SEED=2 bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh
   ```

   批量分发（GPU 池，有空卡即提交下一 job；**脱离 Cursor**）：

   ```bash
   bash experiments/tworoom/scripts/run_dispatch_geometry_retrain_nohup.sh
   # 或直接：scripts/dispatch_geometry_retrain.sh（须 setsid/nohup，勿挂 IDE 终端）
   ```

   > **版本说明（2026-07-13）**：**FT 训练表**、**轨迹偏差 eval 表**、**成功率 eval（实验 1–8）**、**latent kmeanspp predictor FT** 与 **latent kmeanspp 长程成功率 eval** 均为 **`split_seed=3072` + `seed=42`** 正式结果。

脚本（geometry train∩region predictor 已训完；成功率 eval 已完成）：

```bash
# train ∩ geometry region predictors
bash experiments/tworoom/scripts/run_geometry_train_region_predictors.sh

# 可选：全数据 geometry region predictors（轨迹偏差对照）
bash experiments/tworoom/scripts/run_geometry_region_predictors.sh

# 五个 region vs P_train_global（需先训好 region predictors + test cache）
bash experiments/tworoom/scripts/run_geometry_trajectory_deviation.sh
```

等价单条命令：

```bash
python experiments/tworoom/trajectory.py \
  --region-split-mode geometry \
  --restrict-to-train-split \
  --checkpoint /data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt \
  --out-dir experiments/tworoom/results/tworoom_geometry_train_region_predictors \
  --device cuda
```

输出目录：

```bash
experiments/tworoom/results/tworoom_geometry_train_region_predictors/
experiments/tworoom/results/tworoom_geometry_trajectory_predictors/
```

固定几何阈值：`geometry_region_thresholds.npy`（由 `tworoom_geometry_thresholds()` 生成，可复现）

| 名称 | 路径（train∩geometry，训完后） |
|------|------|
| `P_train_common` | `experiments/tworoom/results/tworoom_geometry_train_region_predictors/P_train_common_object.ckpt` |
| `P_train_doorway_corridor` | `.../P_train_doorway_corridor_object.ckpt`（**50ep best**；30ep 用 `P_train_doorway_corridor_epoch30_object.ckpt`） |
| `P_train_left_room` | `.../P_train_left_room_object.ckpt`（**50ep best**；30ep 用 `P_train_left_room_epoch30_object.ckpt`） |
| `P_train_near_wall` | `.../P_train_near_wall_object.ckpt` |
| `P_train_right_room` | `.../P_train_right_room_object.ckpt`（**50ep best**；30ep 用 `P_train_right_room_epoch30_object.ckpt`） |
| `P_train_global_ft` | `.../tworoom_geometry_train_global_ft_65ep/P_train_global_ft_object.ckpt` |
| `P_train_doorway_corridor`（80ep，实验5） | `.../tworoom_geometry_train_region_predictors_doorway80ep/P_train_doorway_corridor_object.ckpt` |

### seed=42 FT 训练完成（2026-07-13）

日志根目录：`results/tworoom_geometry_train_region_predictors/train_{EPOCHS}ep_{region}.log`；global/doorway80/latent 见各 `OUT_DIR/train_*.log`。work 目录：`_work/{region}_{EPOCHS}ep_seed42/`。

**30 ep 收敛（train∩geometry，`seed=42`）**

| region | 样本数 | ep1 train | ep10 | ep20 | ep30 train | ep30 eval | best ep |
|--------|--------|-----------|------|------|------------|-----------|---------|
| common | 297,211 | 0.0162 | 0.00137 | 0.000895 | 0.000713 | **0.000726** | 30 |
| doorway_corridor | 38,552 | 0.0325 | 0.00811 | 0.00526 | 0.00375 | **0.00426** | 30 |
| left_room | 330,243 | 0.0192 | 0.00250 | 0.00155 | 0.00113 | **0.00165** | 30 |
| near_wall | 31,533 | 0.0293 | 0.00319 | 0.00206 | 0.00160 | **0.00155** | 30 |
| right_room | 324,933 | 0.0188 | 0.00251 | 0.00158 | 0.00115 | **0.00110** | 30 |

**50 ep best-by-eval（实验4 三主区，`SELECT_BEST=1`）**

| region | 样本数 | 30ep eval（对照） | 50ep best eval | best ep |
|--------|--------|-------------------|----------------|---------|
| left_room | 330,243 | 0.00165 | **0.000527** | 48 |
| right_room | 324,933 | 0.00110 | **0.000525** | 49 |
| doorway_corridor | 38,552 | 0.00426 | **0.001836** | 47 |

`left_room` / `right_room` ep40 后进入平台；`doorway_corridor` best 在 ep47（ep50 eval 回升至 0.00389）。

**Doorway 80 ep（实验5 专用目录）**：best epoch=**79**，best eval_loss=**0.000872**（`doorway80ep/`，~7.3 min）。

**Global-FT 65 ep（实验8）**：693,728 transitions，best epoch=**64**，best eval_loss=**0.000371**（~97 min；复用 region embedding merge，无 re-encode）。

**Global-FT 50 ep（主实验强 baseline）**：693,728 transitions，best epoch=**47**，best eval_loss=**0.000511**（~77 min；`results/tworoom_geometry_train_global_ft_50ep/`）。

**Latent kmeanspp predictor FT（50 ep，`seed=42`）**

| outer seed | 状态 | cluster0 best eval | cluster1 | cluster2 |
|------------|------|--------------------|----------|----------|
| 0 | ✅ | 0.00110 @ ep40 | 0.000878 @ ep48 | 0.000876 @ ep43 |
| 1 | ✅ | 0.000830 @ ep43 | 0.000910 @ ep47 | 0.000998 @ ep41 |
| 2 | ✅ | 0.000923 @ ep49 | 0.000717 @ ep48 | 0.000813 @ ep41 |

输出：`results/tworoom_latent_kmeanspp_kmeanspp_R50_outer{0,1,2}/P_train_cluster{k}_object.ckpt`。

> **Legacy quantile train∩region 实验**已停掉且未完成；`tworoom_train_region_predictors/` 中的 partial 结果勿用。multipredictor jepa 统一用 geometry。

## 轨迹偏差 eval（seed=42 ckpt，2026-07-13）

与前文 quantile-region 总表同协议：完整 test pool（32517 条），max_steps=10，encoder 用全局权重。predictor 用实验 5 配置：`common`/`near_wall` 30ep、`left_room`/`right_room` 50ep best、`doorway_corridor` 80ep best（`seed=42` FT 权重）。

```bash
bash experiments/tworoom/scripts/run_geometry_train_trajectory_deviation_exp5.sh
```

结果目录：`experiments/tworoom/results/tworoom_geometry_train_trajectory_deviation_{region}_vs_train_exp5/`

### pairwise 偏差（region predictor vs `P_train_global` rollout latent MSE）

| step | `P_train_common` | `P_train_doorway_corridor` | `P_train_left_room` | `P_train_near_wall` | `P_train_right_room` |
|------|------------------|----------------------------|---------------------|---------------------|----------------------|
| 1 | 0.068 | 0.080 | 0.215 | 0.069 | 0.211 |
| 2 | 0.099 | 0.131 | 0.422 | 0.101 | 0.406 |
| 3 | 0.136 | 0.200 | 0.533 | 0.137 | 0.529 |
| 4 | 0.183 | 0.294 | 0.607 | 0.181 | 0.606 |
| 5 | 0.239 | 0.405 | 0.669 | 0.231 | 0.669 |
| 6 | 0.301 | 0.523 | 0.731 | 0.285 | 0.730 |
| 7 | 0.368 | 0.633 | 0.791 | 0.339 | 0.791 |
| 8 | 0.434 | 0.731 | 0.852 | 0.390 | 0.853 |
| 9 | 0.494 | 0.812 | 0.912 | 0.436 | 0.913 |
| 10 | 0.549 | 0.879 | 0.974 | 0.475 | 0.973 |

### 推理误差  MSE（同一次评估附带）

| step | `P_train_common` | `P_train_doorway_corridor` | `P_train_left_room` | `P_train_near_wall` | `P_train_right_room` | `P_train_global` |
|------|------------------|----------------------------|---------------------|---------------------|----------------------|------------------|
| 1 | 0.010 | 0.032 | 0.158 | 0.024 | 0.153 | 0.084 |
| 2 | 0.036 | 0.072 | 0.353 | 0.076 | 0.340 | 0.161 |
| 3 | 0.081 | 0.132 | 0.464 | 0.160 | 0.464 | 0.272 |
| 4 | 0.145 | 0.210 | 0.546 | 0.271 | 0.548 | 0.414 |
| 5 | 0.225 | 0.298 | 0.624 | 0.399 | 0.623 | 0.572 |
| 6 | 0.317 | 0.386 | 0.702 | 0.530 | 0.700 | 0.729 |
| 7 | 0.414 | 0.469 | 0.779 | 0.652 | 0.777 | 0.871 |
| 8 | 0.509 | 0.545 | 0.857 | 0.762 | 0.854 | 0.990 |
| 9 | 0.601 | 0.613 | 0.938 | 0.851 | 0.933 | 1.080 |
| 10 | 0.680 | 0.673 | 1.014 | 0.923 | 1.009 | 1.146 |

### 解读

1. 与旧 quantile-region 总表相比，train∩geometry 版的 pairwise 偏差整体更小（如 `common` step10：0.549 vs 0.846），说明 train-split 上 finetune 出的 predictor 与全局 predictor 更接近。
2. 对 GT 的 rollout MSE：`common` / `doorway_corridor` 在**所有步数**上都优于 `P_train_global`（doorway step10：0.673 vs 1.146，几乎减半）；`near_wall` 在 step≤6 占优。这是全 test pool 平均——即使包含大量区域外样本，这三个 predictor 的长程 rollout 仍更准。
3. `left_room` / `right_room` 在前几步差于 global（step1：0.158/0.153 vs 0.084），因为全 pool 中一半样本在对侧房间（区域外）；step≥7 后追平或略优（step10：1.014/1.009 vs 1.146）。若要干净结论仍需按 test∩region 分层。

> **注意**：上表 `P_train_global` 对照的是**官方未微调** `lewm_object.ckpt` predictor，**不是** Global-FT 50ep。它无法解释为何 region-switch 与 Global-FT 50ep 成功率接近（58–59%）。因此补做了下方四模型同口径动态路由 rollout 对比。

### 四模型动态路由 rollout MSE（vs Global-FT 50ep，2026-07-14；历史 anchor 结果）

**协议说明**：下表由旧实现生成，`history=3` 时按 `history-1` 路由，与用户指定的 rollout 起点 index 0 不一致，因此只保留作历史诊断，不能作为修复后最终对比。当前代码已统一为：`mpc` 固定使用窗口 index 0；`step` 第一次使用 index 0，之后只使用上一时刻模型预测的 latent；未来 GT 仅允许在显式 `oracle_gt_step` 模式中使用。区域分层也改为 index 0。修复后的四模型表需要重新运行后再替换下列数字。

```bash
bash experiments/tworoom/scripts/run_trajectory_switch_rollout_4way.sh
```

汇总：`results/tworoom_trajectory_switch_rollout_4way/rollout_mse_4way.csv`、`.json`、`rollout_mse_delta_vs_global_ft_50ep.csv`

**整体 rollout MSE vs GT**

| step | Official LeWM | Global-FT 50ep | rooms3 switch 50ep | latent cluster-switch |
|------|--------------:|---------------:|-------------------:|----------------------:|
| 1 | 0.084 | **0.006** | 0.006 | 0.007 |
| 3 | 0.272 | **0.072** | 0.070 | 0.074 |
| 5 | 0.572 | 0.216 | **0.210** | 0.218 |
| 10 | 1.146 | 0.679 | **0.666** | 0.681 |

**相对 Global-FT 50ep 的 ΔMSE（负=更好）**

| step | rooms3 switch | latent cluster-switch |
|------|--------------:|----------------------:|
| 1 | −0.00005 | +0.00082 |
| 3 | −0.0017 | +0.0021 |
| 5 | −0.0053 | +0.0030 |
| 10 | **−0.012** | +0.0027 |

step10 轨迹级：rooms3 57.4% 样本 MSE 低于 Global-FT，latent 45.6%（其余接近打平）。

**解读（对应判断标准）**

1. **Official → Global-FT**：step10 MSE 从 1.146 降至 0.679（−41%），与 baseline 49.2% → Global-FT 58.4% 的成功率跃升一致——**后训练是主要增益来源**。
2. **Global-FT vs region/cluster switch**：四者 step10 MSE 均在 0.66–0.68，差距 <2%。与长程成功率（Global-FT 58.4%、rooms3 59.2%、cluster 59.6%）同量级。**分区/聚类路由未带来决定性额外预测优势**；成功率 +0.4–1.2pp 更像 Global-FT 已进入“足够准”区间后的噪声/路由边界效应。
3. **rooms3 长 rollout 略优**：step10 Δ=−0.012，与“left/right 专家前几步差、长程反超”的固定专家曲线一致；但优势很小，尚不足以单独支撑“任务精度门槛”叙事。
4. **二值成功率信息损失**：此离线实验直接量化 latent 误差；后续控制 eval 应补最终距离、达标步数、多阈值成功率、与 Global-FT 的 per-episode 成败翻转。

##实验
在two room数据集上跑任务成功率实验
1.baseline le-wm
2.完全一样的设置下，根据每一步推理的起点从`P_train_left_room`、`P_train_doorway_corridor`、`P_train_right_room`选择对应的predictor
3.完全一样的设置下，根据每一步推理的起点从`P_train_left_room`、`P_train_doorway_corridor`、`P_train_right_room`、`P_train_near_wall`、 `P_train_common`
优先级：
  1.`P_train_doorway_corridor`
  2.`P_train_near_wall`
  3.`P_train_common`
  4.`P_train_right_room`、`P_train_left_room`
4.将 predictor 在三个主区域上从训练 30 轮增加到 **50 轮**（`doorway_corridor`、`left_room`、`right_room`；`common` / `near_wall` 仍用 30 轮版本）。50 轮内每 epoch 记 eval_loss；从 ep20 起每 10 轮存 checkpoint；最终 `P_train_*_object.ckpt` 取 **eval_loss 最小** epoch（非 last epoch）。——较baseline耗时约1.2h

   训练收敛（train∩geometry，50 ep，`seed=42`）：

   | region | 样本数 | 30ep eval | 50ep best eval_loss | best epoch |
   |--------|--------|-----------|---------------------|------------|
   | left_room | 330,243 | 0.00165 | **0.000527** | 48 |
   | right_room | 324,933 | 0.00110 | **0.000525** | 49 |
   | doorway_corridor | 38,552 | 0.00426 | **0.001836** | 47 |

   `left_room` / `right_room` 在 ep40 后进入平台；`doorway_corridor` best 在 ep47（末 ep eval 略升）。

   重复实验 2、3（rooms3 / priority5），5 seeds，复用各 seed baseline 的 eval 起点（`seed=42` ckpt，2026-07-13）。

   ```bash
   bash experiments/tworoom/scripts/run_geometry_train_region_predictors_50ep_best.sh
   bash experiments/tworoom/scripts/run_success_rate_5seed_50ep_rooms3_priority5.sh
   ```

   汇总：`experiments/tworoom/results/tworoom_success_rate_50ep_5seed_summary.csv`、`.json`
5.在实验4的基础上，doorway_corridor 训练轮次从50轮增加到80轮以保证完全收敛
  保存到 `experiments/tworoom/results/tworoom_geometry_train_region_predictors_doorway80ep/P_train_doorway_corridor_object.ckpt`（80 ep 内 eval_loss 最小 epoch=**79**，best eval_loss=**0.000872**，`seed=42`）
  在实验4的基础上替换 P_train_doorway_corridor，进行成功率评估（rooms3 + priority5，5 seeds）（`seed=42` ckpt，2026-07-13）。

  ```bash
  bash experiments/tworoom/scripts/run_success_rate_5seed_exp5_priority5.sh
  ```

  汇总：`experiments/tworoom/results/tworoom_success_rate_exp5_5seed_summary.csv`、`.json`
6.长程实验，将推理成功率配置从goal_offset=25, eval_budget=50改成goal_offset=50, eval_budget=50。重测总表全部配置（baseline + 实验2/3/4/5）

  baseline 用 goal_offset=50 重新采样各 seed 的 `eval_start_indices`；其余配置复用同 seed 的 exp6 baseline 起点。30ep 使用 `P_train_*_epoch30`（doorway/left/right）；50ep 使用 geometry_train 目录 best；实验5 额外替换 80ep doorway。

  ```bash
  bash experiments/tworoom/scripts/run_success_rate_5seed_exp6_longrange.sh
  bash experiments/tworoom/scripts/run_success_rate_5seed_exp6_longrange_30ep_50ep.sh
  ```

  汇总：`tworoom_success_rate_exp6_5seed_summary.csv`（baseline + 实验5）、`tworoom_success_rate_exp6_30ep_50ep_5seed_summary.csv`（实验2/4）

  **长程成功率（goal_offset=50，`seed=42` ckpt，2026-07-13）**

  | seed | baseline | 实验2 rooms3 (30ep) | 实验3 priority5 (30ep) | 实验4 rooms3 (50ep) | 实验4 priority5 (50ep) | 实验5 rooms3 (80ep doorway) | 实验5 priority5 (80ep doorway) | 实验8 global_ft (65ep) |
  |------|----------|---------------------|------------------------|---------------------|------------------------|----------------------------|----------------------------------|-------------------------|
  | 0 | 50.0% (25/50) | 60.0% (30/50) | 60.0% (30/50) | 60.0% (30/50) | 60.0% (30/50) | 60.0% (30/50) | 60.0% (30/50) | 60.0% (30/50) |
  | 1 | 56.0% (28/50) | 68.0% (34/50) | 66.0% (33/50) | 64.0% (32/50) | 64.0% (32/50) | 66.0% (33/50) | 66.0% (33/50) | 62.0% (31/50) |
  | 2 | 42.0% (21/50) | 56.0% (28/50) | 56.0% (28/50) | 60.0% (30/50) | 58.0% (29/50) | 60.0% (30/50) | 58.0% (29/50) | 52.0% (26/50) |
  | 3 | 48.0% (24/50) | 58.0% (29/50) | 60.0% (30/50) | 54.0% (27/50) | 56.0% (28/50) | 56.0% (28/50) | 58.0% (29/50) | 52.0% (26/50) |
  | 4 | 50.0% (25/50) | 58.0% (29/50) | 60.0% (30/50) | 58.0% (29/50) | 58.0% (29/50) | 58.0% (29/50) | 58.0% (29/50) | 60.0% (30/50) |
  | 均值 ± std | 49.2 ± 5.0% | 60.0 ± 4.7% | 60.4 ± 3.6% | 59.2 ± 3.6% | 59.2 ± 3.0% | 60.0 ± 3.7% | 60.0 ± 3.5% | 57.2 ± 4.8% |

  长程下 region-switch 配置均显著高于 baseline（约 +11pp）；30ep / 50ep / 80ep doorway 均值几乎相同（~60%）；global_ft（57.2%）低于 region-switch 但高于 baseline。

7.推理速度基准：region-switch 相对 baseline 的推理开销

  计时对象为一次完整 replan（50 个 env 并行、horizon=5、CEM 300 candidates × 30 iters，即"推理50次×5步"），实验 5 权重，seed 0 起点，重复 3 次取首次 replan（三种 mode 工作量完全相同）。

  ```bash
  python experiments/tworoom/benchmark_inference_speed.py --seed 0 --repeats 3
  ```

  | mode | 50-env replan 耗时 | 单 env 单次 replan | vs baseline |
  |------|--------------------|--------------------|-------------|
  | baseline | 44.67 ± 0.71 s | 893 ms | — |
  | rooms3 | 45.80 ± 0.33 s | 916 ms | **+2.5%** |
  | priority5 | 46.26 ± 0.37 s | 925 ms | **+3.5%** |

  优化：将几何阈值预先转换到标准化 proprio 坐标，移除每次选择 predictor 时的 GPU→CPU→NumPy→sklearn 反标准化；同一个 CEM batch 的 30 次 `get_cost` 复用 region 选择结果。优化后 rooms3 / priority5 的开销由约 6% 降至 2.5% / 3.5%。predictor 本身结构与 baseline 完全一致，前向 FLOPs 不变；优化前后 seed 0 的逐 episode 结果完全一致（0 flip）。

  结果：`experiments/tworoom/results/tworoom_inference_speed_benchmark.json`

评估协议：`config/eval/tworoom.yaml`（50 episodes，`goal_offset=25`，`eval_budget=50`，CEM）；seeds `0–4`；每个 seed 的实验 2/3/4/5 复用同 seed baseline 的 `eval_start_indices`。实验 2/3 短程 30ep 使用显式 `P_train_*_epoch30_object.ckpt` override。

```bash
bash experiments/tworoom/scripts/run_success_rate_5seed.sh
```

汇总：`experiments/tworoom/results/tworoom_success_rate_5seed_summary.csv`、`tworoom_success_rate_5seed_summary.json`（30ep 见 `tworoom_success_rate_30ep_5seed_summary.csv`）

**短程成功率（goal_offset=25，`seed=42` ckpt，2026-07-13）**

| seed | 实验1 baseline | 实验2 rooms3 (30ep) | 实验3 priority5 (30ep) | 实验4 rooms3 (50ep) | 实验4 priority5 (50ep) | 实验5 rooms3 (80ep doorway) | 实验5 priority5 (80ep doorway) | 实验8 global_ft (65ep) |
|------|----------------|---------------------|------------------------|---------------------|------------------------|----------------------------|----------------------------------|-------------------------|
| 0 | 90.0% (45/50) | 96.0% (48/50) | 96.0% (48/50) | 98.0% (49/50) | 96.0% (48/50) | 98.0% (49/50) | 96.0% (48/50) | 96.0% (48/50) |
| 1 | 94.0% (47/50) | 94.0% (47/50) | 94.0% (47/50) | 94.0% (47/50) | 94.0% (47/50) | 94.0% (47/50) | 94.0% (47/50) | 94.0% (47/50) |
| 2 | 84.0% (42/50) | 86.0% (43/50) | 86.0% (43/50) | 88.0% (44/50) | 88.0% (44/50) | 88.0% (44/50) | 88.0% (44/50) | 86.0% (43/50) |
| 3 | 96.0% (48/50) | 94.0% (47/50) | 92.0% (46/50) | 92.0% (46/50) | 92.0% (46/50) | 92.0% (46/50) | 92.0% (46/50) | 92.0% (46/50) |
| 4 | 88.0% (44/50) | 92.0% (46/50) | 94.0% (47/50) | 92.0% (46/50) | 94.0% (47/50) | 92.0% (46/50) | 94.0% (47/50) | 94.0% (47/50) |
| 均值 ± std | 90.4 ± 4.8% | 92.4 ± 3.8% | 92.4 ± 4.3% | 92.8 ± 3.6% | 92.8 ± 3.0% | 92.8 ± 3.6% | 92.8 ± 3.0% | 92.4 ± 3.8% |

短程 region-switch 各配置均高于 baseline（+2.0~2.4pp）；30ep / 50ep / 80ep doorway 差别很小。
8.由于最高指标的模型priority5 (50ep)等价于全局微调64.2轮，由于我们在所有区域上的recipe除训练轮次外完全一致，所以我们在训练集用同样的recipe微调训练predictor 65轮得到 **Global-FT compute-matched**（单 predictor、无 region switch）。

   训练配置（与 region FT 相同，仅数据为全 train split）：
   `epochs=65, lr=5e-5, batch_size=128, weight_decay=1e-3, history_size=3`；冻结 encoder；`select_best_by_eval`；**仅保存 eval_loss 最小的 best model**（`P_train_global_ft_object.ckpt`，不存中间 epoch checkpoint）。
   **编码复用**：不重新跑 encoder；从已有 `P_train_{left,doorway,right}_embeddings.npz` 合并得到全 train split（693,728 transitions，已验证与 `train_global_reference_starts.npy` 一致）。

   ```bash
   # 1) 训练 Global-FT（~1.5h，可 detached）
   bash experiments/tworoom/scripts/run_geometry_train_global_ft_65ep_nohup.sh

   # 2) 短程成功率（goal_offset=25，5 seeds，复用 baseline eval_start_indices）
   bash experiments/tworoom/scripts/run_success_rate_5seed_exp8_global_ft.sh

   # 3) 长程成功率（goal_offset=50，5 seeds，复用 exp6 baseline eval_start_indices）
   bash experiments/tworoom/scripts/run_success_rate_5seed_exp8_global_ft_longrange.sh
   ```

   **训练完成**（693,728 transitions，best epoch=**64**/65，best eval_loss=**0.000371**，`seed=42`）  
   日志：`results/tworoom_geometry_train_global_ft_65ep/train_global_ft_65ep.log`（总耗时 ~97 min）

   ### 短程结果（goal_offset=25，5 seeds，`seed=42` ckpt，2026-07-13）

   | seed | 实验8 global_ft (65ep) |
   |------|-------------------------|
   | 0 | 96.0% (48/50) |
   | 1 | 94.0% (47/50) |
   | 2 | 86.0% (43/50) |
   | 3 | 92.0% (46/50) |
   | 4 | 94.0% (47/50) |
   | 均值 ± std | **92.4 ± 3.8%** |

   汇总：`results/tworoom_success_rate_exp8_global_ft_5seed_summary.csv`、`.json`

   ### 长程结果（goal_offset=50，5 seeds，`seed=42` ckpt，2026-07-13）

   | seed | 实验8 global_ft (65ep) |
   |------|-------------------------|
   | 0 | 60.0% (30/50) |
   | 1 | 62.0% (31/50) |
   | 2 | 52.0% (26/50) |
   | 3 | 52.0% (26/50) |
   | 4 | 60.0% (30/50) |
   | 均值 ± std | **57.2 ± 4.8%** |

   汇总：`results/tworoom_success_rate_exp8_global_ft_exp6_5seed_summary.csv`、`.json`

   ### 结论（compute-matched 对照，`seed=42` ckpt，2026-07-13）

   | 协议 | baseline | 实验8 global_ft (65ep) | 实验4 priority5 (50ep) |
   |------|----------|-------------------------|-------------------------|
   | 短程 (goal_offset=25) | 90.4 ± 4.8% | 92.4 ± 3.8% | **92.8 ± 3.0%** |
   | 长程 (goal_offset=50) | 49.2 ± 5.0% | 57.2 ± 4.8% | **59.2 ± 3.0%** |

   1. **短程**：global_ft 与 region-switch 几乎持平（92.4% vs 92.8%），均高于 baseline（+2.0~2.4pp）。
   2. **长程**：region-switch ~60% 显著高于 baseline（+11pp）；global_ft 57.2% 介于两者之间；训练轮次间差别很小。

   ### 微调随机种子稳定性（train seeds = 0/42/625，2026-07-15）

   为区分**训练随机性**与**测试任务随机性**，rooms3、priority5 和 Global-FT65 均补测 predictor fine-tuning seeds `0/42/625`。每个 train seed 仍在完全相同的 5 个 eval seeds（`0–4`）和相同 `eval_start_indices` 上评测：先对每个 train seed 的 5 个 eval 结果求均值，再对三个 train-seed 均值计算 sample SD。所以下表的 `±` 只表示**微调随机性**，不再表示 eval-seed 波动。

   | 协议 | 模型 | train seed 0 | train seed 42 | train seed 625 | train-seed mean ± SD |
   |------|------|-------------:|--------------:|---------------:|---------------------:|
   | 短程 | rooms3 30ep | 92.0% | 92.8% | 91.2% | **92.0 ± 0.8%** |
   | 短程 | priority5 30ep | 92.0% | 92.8% | 91.2% | **92.0 ± 0.8%** |
   | 短程 | rooms3 50ep | 91.6% | 92.8% | 91.2% | **91.9 ± 0.8%** |
   | 短程 | priority5 50ep | 92.0% | 92.8% | 91.2% | **92.0 ± 0.8%** |
   | 短程 | rooms3 doorway80 | 92.0% | 92.8% | 91.2% | **92.0 ± 0.8%** |
   | 短程 | priority5 doorway80 | 91.6% | 92.8% | 91.6% | **92.0 ± 0.7%** |
   | 短程 | Global-FT65 | 89.2% | 92.4% | 89.6% | **90.4 ± 1.7%** |
   | 长程 | rooms3 30ep | 61.2% | 60.0% | 62.4% | **61.2 ± 1.2%** |
   | 长程 | priority5 30ep | 61.6% | 60.4% | 62.8% | **61.6 ± 1.2%** |
   | 长程 | rooms3 50ep | 58.8% | 59.2% | 61.6% | **59.9 ± 1.5%** |
   | 长程 | priority5 50ep | 58.8% | 59.2% | 63.2% | **60.4 ± 2.4%** |
   | 长程 | rooms3 doorway80 | 58.8% | 60.0% | 62.0% | **60.3 ± 1.6%** |
   | 长程 | priority5 doorway80 | 59.2% | 60.0% | 63.2% | **60.8 ± 2.1%** |
   | 长程 | Global-FT65 | 58.0% | 57.2% | 58.0% | **57.7 ± 0.5%** |

   **Compute-matched 结论**：priority5 50ep 相对 Global-FT65，短程由 **90.4 ± 1.7%** 提升到 **92.0 ± 0.8%**（+1.6pp），长程由 **57.7 ± 0.5%** 提升到 **60.4 ± 2.4%**（+2.7pp）；三个 train seeds 下 region-switch 均保持更高均值。因此 seed42 的优势不是单次微调初始化偶然造成的。与此同时，priority5 50ep 的长程 train-seed SD 更大，且当前只有 3 个 train seeds，应作为稳定性证据而非显著性检验。

   **轮次观察**：30ep 的长程结果最高（rooms3 **61.2 ± 1.2%**、priority5 **61.6 ± 1.2%**），没有观察到增加到 50ep/doorway80 后单调提升。

   Baseline 没有 predictor fine-tuning seed，因此不参与 train-seed SD；更新后的成功率图只把 baseline 作为固定参考值（短程 90.4%、长程 49.2%），不会把 eval-seed SD 与 train-seed SD 混在同一误差棒中。

   汇总：
   - `results/tworoom_success_rate_geometry_global65_finetune_seed_summary.csv`（跨 train-seed mean ± sample SD）
   - `results/tworoom_success_rate_geometry_global65_per_finetune_seed.csv`（每个 train seed 的 5-eval-seed 明细）
   - `results/tworoom_success_rate_geometry_global65_finetune_seed_summary.json`

   输出：
   - 权重：`results/tworoom_geometry_train_global_ft_65ep/P_train_global_ft_object.ckpt`
   - 评估：`results/tworoom_success_rate_global_ft_65ep_seed{0,1,2,3,4}/`（短程）
   - 评估：`results/tworoom_success_rate_global_ft_65ep_exp8_exp6_seed{0,1,2,3,4}/`（长程）

   评估时用 `--mode baseline --checkpoint P_train_global_ft_object.ckpt`（单 predictor，与实验1 协议相同，仅替换权重）。

结果目录：

- `experiments/tworoom/results/tworoom_success_rate_baseline_seed{0,1,2,3,4}/`
- `experiments/tworoom/results/tworoom_success_rate_rooms3_seed{0,1,2,3,4}/`（实验 2，30ep predictor）
- `experiments/tworoom/results/tworoom_success_rate_priority5_seed{0,1,2,3,4}/`（实验 3，30ep predictor）
- `experiments/tworoom/results/tworoom_success_rate_rooms3_50ep_seed{0,1,2,3,4}/`（实验 4）
- `experiments/tworoom/results/tworoom_success_rate_priority5_50ep_seed{0,1,2,3,4}/`（实验 4）
- `experiments/tworoom/results/tworoom_success_rate_rooms3_exp5_seed{0,1,2,3,4}/`（实验 5）
- `experiments/tworoom/results/tworoom_success_rate_priority5_exp5_seed{0,1,2,3,4}/`（实验 5）
- `experiments/tworoom/results/tworoom_geometry_train_region_predictors/`（含 50ep best + epoch20/30/40/50 checkpoint）
- `experiments/tworoom/results/tworoom_geometry_train_region_predictors_doorway80ep/`（实验 5，80ep doorway best）
- `experiments/tworoom/results/tworoom_geometry_train_global_ft_65ep/`（实验 8，Global-FT compute-matched）
- `experiments/tworoom/results/tworoom_success_rate_global_ft_65ep_seed{0,1,2,3,4}/`（实验 8 短程）
- `experiments/tworoom/results/tworoom_success_rate_global_ft_65ep_exp8_exp6_seed{0,1,2,3,4}/`（实验 8 长程）
- `experiments/tworoom/results/tworoom_success_rate_{baseline,rooms3,priority5}_exp6_seed{0,1,2,3,4}/`（实验 6，goal_offset=50，baseline + 实验5）
- `experiments/tworoom/results/tworoom_success_rate_{rooms3,priority5}_exp6_{30ep,50ep}_seed{0,1,2,3,4}/`（实验 6，实验2/4 长程） 
# 自动划分实验
## rooms3
用 **linear probe** 检验 rooms3 三区是否可从 LeWM 潜表示中可区分。注意：实现**不是** hinge-loss SVM，而是 PyTorch 上的 softmax 线性探针及其 RFF 扩展。

**样本单位：单个编码潜向量**（192-d），按 **global timestep 去重**（滑动窗口展开后同一时刻只保留一条；2,774,912 展开 → **909,723** 唯一时刻）。每条向量用**该时刻** agent 的 geometry 位置打 rooms3 标签。

**划分单位：episode**（`ep_idx`）。在 P_train 缓存覆盖的潜向量上，按 episode 随机切分（`episode_split_seed=20260711`，70% / 15% / 15% train/val/test），保证三个集合来自**不相交的 episode**。探针只在 train episode 的潜向量上拟合，在 val / test episode 上评估。

| 探针 | 实现 |
|------|------|
| **Linear Softmax Probe** | `nn.Linear(192,3)` + `CrossEntropyLoss` + `AdamW`（PyTorch GPU） |
| **RFF-RBF Probe** | Random Fourier Features（RBF 核近似）+ 线性 softmax 分类头（PyTorch GPU） |

```bash
bash experiments/tworoom/scripts/run_geometry_latent_svm_rooms3.sh
```

脚本：`experiments/tworoom/geometry_latent_svm_rooms3.py`（历史文件名保留 `svm` 字样）  
日志：`results/geometry_latent_svm_rooms3/run_full_cached.log`（去重后总耗时 ~63s，5 epoch，GPU `cuda`）

## priority5
将 priority5 的划分区域转化成**互斥**五类（与 MPC `geometry_priority5_key` 相同的优先级级联）：

1. **doorway_corridor**（doorway）
2. **near_wall** \ doorway
3. **common** \ (doorway ∪ near_wall)
4. **right_room** \ (near_wall ∪ common)
5. **left_room** \ (near_wall ∪ common)

执行与 rooms3 **相同协议**：单个 192-d 潜向量样本、global timestep 去重、episode-level 70/15/15 划分、Linear Softmax Probe + RFF-RBF Probe（PyTorch GPU，5 epoch）。

| 探针 | 实现 |
|------|------|
| **Linear Softmax Probe** | `nn.Linear(192,5)` + `CrossEntropyLoss` + `AdamW`（PyTorch GPU） |
| **RFF-RBF Probe** | Random Fourier Features（RBF 核近似）+ 线性 softmax 分类头（PyTorch GPU） |

```bash
bash experiments/tworoom/scripts/run_geometry_latent_probe_priority5.sh
```

脚本：`experiments/tworoom/geometry_latent_svm_rooms3.py --partition priority5`  
日志：`results/geometry_latent_probe_priority5/run_full_cached.log`（去重后总耗时 ~71s，5 epoch，GPU `cuda`）

### 已完成结果：geometry priority5 潜向量线性可区分性（episode split + timestep dedup）

数据：五区 P_train 缓存展开 4,089,888 条 → priority5 互斥标签后按 global timestep 去重为 **909,723** 条唯一潜向量（去掉 3,180,165 条窗口重叠重复）。episode 划分 7000 / 1500 / 1500，对应 **637,122 / 136,168 / 136,433** 条向量。类别比例：common ~67%，left/right ~23% 各，doorway ~5.5%，near_wall ~4%。

#### 总表（test episode 潜向量）

| 探针 | accuracy | balanced acc | macro-F1 | doorway F1 | near_wall F1 | common F1 | right F1 | left F1 |
|------|----------|--------------|----------|------------|--------------|-----------|----------|---------|
| Linear Softmax Probe | 98.69% | 98.04% | **97.96%** | 99.18% | 94.29% | 99.16% | 98.55% | 98.61% |
| RFF-RBF Probe | 99.20% | 98.80% | **98.77%** | 99.48% | 96.64% | 99.49% | 99.16% | 99.09% |

#### Val episode（一致性检查）

| 探针 | accuracy | balanced acc | macro-F1 | near_wall F1 |
|------|----------|--------------|----------|--------------|
| Linear Softmax Probe | 98.75% | 98.10% | 98.07% | 94.66% |
| RFF-RBF Probe | 99.24% | 98.83% | 98.85% | 96.81% |

#### 结论

1. **priority5 互斥五区同样高度线性可区分**：Linear Softmax Probe test macro-F1 **97.96%**（较 rooms3 的 99.60% 略低，符合更细粒度划分预期）。
2. **near_wall 相对最弱**（test F1 ~94.3–96.6%），与其样本占比最低（~4%）及与 doorway/common 边界过渡带一致。
3. RFF-RBF 对 near_wall 有约 **2pp** 提升（94.29% → 96.64%），说明少量非线性边界仍有帮助。
4. episode-level holdout + timestep 去重下仍近 98% macro-F1，支持在潜空间做 priority5 级联区域划分。

#### 5-seed 稳定性检验（seed = 0,1,2,3,4）

三类随机性由**同一 seed** 联合控制：episode 划分、Linear Probe 初始化/minibatch 顺序、RFF 随机傅里叶特征 \(\omega_i,b_i\)。同一 seed 下 Linear 与 RFF-RBF 使用完全相同的 episode split。

```bash
bash experiments/tworoom/scripts/run_geometry_latent_probe_priority5_multiseed.sh
```

日志：`results/geometry_latent_probe_priority5_multiseed/run_multiseed.log`（5 seeds × 2 probes，~239s）

**Test episode 汇总（mean ± std，5 seeds）**

| 探针 | accuracy | balanced acc | macro-F1 | doorway F1 | near_wall F1 | common F1 | right F1 | left F1 |
|------|----------|--------------|----------|------------|--------------|-----------|----------|---------|
| Linear Softmax Probe | 98.70% ± 0.06% | 98.11% ± 0.05% | **98.01% ± 0.08%** | 99.08% ± 0.08% | 94.60% ± 0.24% | 99.14% ± 0.06% | 98.58% ± 0.08% | 98.64% ± 0.07% |
| RFF-RBF Probe | 99.21% ± 0.03% | 98.80% ± 0.11% | **98.79% ± 0.05%** | 99.36% ± 0.08% | 96.81% ± 0.14% | 99.49% ± 0.04% | 99.11% ± 0.05% | 99.17% ± 0.05% |

**配对差值** \(\Delta_s = \mathrm{MacroF1}_{\mathrm{RFF},s} - \mathrm{MacroF1}_{\mathrm{Linear},s}\)：

| 统计量 | 值 |
|--------|-----|
| mean ± std | **+0.78pp ± 0.05pp** |
| per-seed | +0.74, +0.79, +0.73, +0.88, +0.75 pp |
| RFF > Linear | **5 / 5 seeds** |

单 seed（`model_seed=20260711`）的 +0.81pp 非线性提升与 5-seed 均值 +0.78pp 一致；RFF 在所有 seed 上均优于 Linear，**priority5 的非线性边界增益是稳定的**，尤其体现在 near_wall（linear 94.60% → RFF 96.81%，+2.2pp）。

输出文件：

- `experiments/tworoom/results/geometry_latent_probe_priority5/geometry_latent_probe_priority5.json`
- `experiments/tworoom/results/geometry_latent_probe_priority5/geometry_latent_probe_priority5_metrics.csv`
- `experiments/tworoom/results/geometry_latent_probe_priority5/geometry_latent_probe_priority5_split.npz`
- `experiments/tworoom/results/geometry_latent_probe_priority5_multiseed/geometry_latent_probe_priority5_multiseed.json`
- `experiments/tworoom/results/geometry_latent_probe_priority5_multiseed/geometry_latent_probe_priority5_multiseed_summary.csv`

## 主实验

对训练集潜向量（去重后）做无监督 3 类划分，按类微调 predictor，再根据潜向量路由到对应 predictor，测长程成功率。控制评测同时保留两种路由协议：历史的**逐 MPC 路由**，以及当前预期方法的 **rollout 内逐预测步动态路由**。

**聚类阶段已结论**（见下文 K-means++ 多重启）：论文主配置为 `zscore_l2` + spherical K-means++，`n_init=50`；**不再增加 restart**。下游 predictor 实验固定 **outer seeds 0、1、2** 的 \(R=50\) 划分。

### 协议

1. **聚类数据**：从 `P_train_{doorway,near_wall,common,right,left}_embeddings.npz` 展开 sliding window 后按 global timestep 去重 → **909,723** 条唯一 192-d 潜向量。
2. **聚类方法（论文主配置）**：`zscore_l2` + spherical K-means++；`K=3`；`n_init=50`；`max_iter=1000`；`rel_tol=1e-7`；`patience=10`；选 maximum objective。产物：`labels/kmeanspp_R50_outer{0,1,2}.npz` + `zscore_params.npz`。
3. **按类微调**：以 transition 起点 `global_idx` 查表得 cluster；在 693,728 条 train transitions 上每类独立 FT predictor **50 ep**，`select_best_by_eval`，**`--seed 42`**（固定 DataLoader shuffle）→ `P_train_cluster{k}_object.ckpt`。**每个 outer seed 各跑一套**。
4. **强 baseline（Global-FT 50ep）**：同一 train split、同一 recipe（`lr=5e-5, batch=128, weight_decay=1e-3, history_size=3, seed=42, select_best_by_eval`），**单 predictor、无 cluster 路由**；与每个 cluster predictor 的训练轮次对齐，用于对照「分 cluster 特化 + 潜空间路由」是否优于全局 50ep FT。
5. **长程推理**：`goal_offset=50`，eval **5 seed**，复用 exp6 baseline `eval_start_indices`。cluster 路由均使用 **Z-score（`zscore_params.npz`）→ 最近 centroid**，但分为两种协议：
   - `--latent-routing mpc`（历史对照，默认）：每次 MPC 根据当前真实观测编码出的 latent 选一次 predictor；该次 CEM imagined rollout 内固定 predictor。
   - `--latent-routing step`（当前预期协议）：rollout 第一步根据当前观测 latent 路由；之后每条 CEM candidate 在每个 imagined step 根据上一时刻**预测 latent**重新路由，不使用未来 GT，不同 candidate 可选择不同 predictor。
   - 人工 `rooms3` / `priority5` 继续根据当前 proprio 和几何规则逐 MPC 路由，不纳入逐预测步补测。
6. **指标**：聚类耗时（见 K-means++ 效率表）、`success_rate`；路由延迟统一记录为 `inference_classify_per_call_ms` 和 `inference_classify_per_assignment_us`（CUDA event 采样，最多 2048 次调用），逐预测步版本另记录 route calls、candidate-step assignments、cluster histogram 与 rollout 内 route switch rate。
7. **汇报**：cluster 配置报 **总均值**、**eval-seed SD**（5 个 eval seed，每个 seed 对 3 个 outer 取均后再求 SD）、**outer-seed SD**（3 个 outer 各自 5-seed 均值的 SD）；其余对照（baseline / rooms3 50ep / Global-FT 50ep）仅单划分，报 **均值 ± eval-seed SD**（5 eval seed）。**不可**把 cluster 的 outer-seed SD 与其他方法的 eval-seed SD 混在同一 ± 里比较。

历史基线（FAISS CPU 3-seed，ARI≈0.05）仍保留在 `results/latent_unsup_cluster/`，仅作对照，**不作为论文主配置**。

### 运行命令（predictor 下游）

```bash
# Step 1: 聚类已完成 → results/latent_kmeanspp_multirestart_k3/

# Step 2: 每类 predictor 微调 50ep（每个 outer seed 各跑一次；GPU 必填）
GPU=0 OUTER_SEED=0 bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh
GPU=1 OUTER_SEED=1 bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh
GPU=2 OUTER_SEED=2 bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh

# Step 2b: Global-FT 50ep 强 baseline（单 predictor，与 cluster 50ep 轮次对齐）
bash experiments/tworoom/scripts/run_geometry_train_global_ft_50ep_nohup.sh
# 或：GPU=0 bash experiments/tworoom/scripts/run_geometry_train_global_ft_50ep.sh

# Step 3a: 历史逐 MPC 路由，长程成功率 5-seed eval（每个 outer seed）
OUTER_SEED=0 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_kmeanspp_longrange.sh
OUTER_SEED=1 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_kmeanspp_longrange.sh
OUTER_SEED=2 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_kmeanspp_longrange.sh
# 或一次跑完 outer 0/1/2（脱离 Cursor）：
bash experiments/tworoom/scripts/run_success_rate_latent_kmeanspp_longrange_all_outer_nohup.sh

# Step 3b: rollout 内逐预测步动态路由（每个 outer 5 eval seeds）
OUTER_SEED=0 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_kmeanspp_step_routing_longrange.sh
OUTER_SEED=1 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_kmeanspp_step_routing_longrange.sh
OUTER_SEED=2 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_kmeanspp_step_routing_longrange.sh
# 三个 outer 并行（GPU 编号可覆盖）：
GPU0=0 GPU1=1 GPU2=2 bash experiments/tworoom/scripts/run_success_rate_latent_kmeanspp_step_routing_parallel.sh

# Step 3c: Global-FT 50ep 长程 eval（5 seeds，复用 exp6 baseline starts）
bash experiments/tworoom/scripts/run_success_rate_global_ft_50ep_longrange_nohup.sh
```

### 历史对照：逐 MPC latent 路由

**长程成功率（goal_offset=50，`seed=42` ckpt，2026-07-13/14）**

| eval seed | outer=0 | outer=1 | outer=2 |
|-----------|---------|---------|---------|
| 0 | 60.0% (30/50) | 62.0% (31/50) | 54.0% (27/50) |
| 1 | 68.0% (34/50) | 66.0% (33/50) | 68.0% (34/50) |
| 2 | 58.0% (29/50) | 58.0% (29/50) | 56.0% (28/50) |
| 3 | 60.0% (30/50) | 56.0% (28/50) | 52.0% (26/50) |
| 4 | 58.0% (29/50) | 60.0% (30/50) | 58.0% (29/50) |
| 均值 ± std | **60.8 ± 4.1%** | **60.4 ± 3.8%** | **57.6 ± 6.2%** |

**汇总（3 outer）**：总均值 **59.6%**；eval-seed SD = **4.5%**（各 eval seed 在 3 outer 上的均值之 SD）；outer-seed SD = **1.7%**（outer=0/1/2 各自 5-seed 均值之 SD）。

| 对照 | 均值 | eval-seed SD | 备注 |
|------|-----:|-------------:|------|
| baseline | 49.2% | 5.0% | 单划分，5 eval seed |
| geometry region-switch（**rooms3 50ep**） | 59.2% | 3.6% | 单划分，5 eval seed |
| **latent kmeanspp cluster-switch** | **59.6%** | **4.5%** | outer-seed SD = **1.7%**（3 outer） |
| **Global-FT 50ep**（强 baseline） | 58.4% | 3.0% | 单划分，5 eval seed |

**Global-FT 50ep 逐 seed**

| eval seed | 0 | 1 | 2 | 3 | 4 |
|-----------|---|---|---|---|---|
| success rate | 58.0% (29/50) | 62.0% (31/50) | 58.0% (29/50) | 54.0% (27/50) | 60.0% (30/50) |

逐 MPC 推理路由开销：**~3.3 ms/step**（encode + Z-score + nearest centroid；50-env 并行 replan 内均摊）。该数字不适用于下方逐预测步协议。

汇总：cluster — `results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer{0,1,2}_5seed_summary.csv`；Global-FT 50ep — `results/tworoom_success_rate_global_ft_50ep_exp6_5seed_summary.csv`、`.json`

**结论**：cluster-switch 总均值 59.6%，与 geometry **rooms3 50ep**（59.2%）同量级（+0.4pp）；eval-seed SD（4.5%）与 rooms3（3.6%）相当。相对 Global-FT 50ep（58.4%）略高 +1.2pp。outer-seed SD 仅 1.7pp，三种划分稳定；均显著高于 baseline（49.2%）。**离线四模型 rollout MSE（见上方）显示 Global-FT 已解释绝大部分预测增益；region/cluster 路由的额外 MSE 优势 <2%，与成功率边际一致。**

### 补测：rollout 内逐预测步动态路由（当前预期协议，2026-07-14）

实现：`tworoom_success_rate_eval.py --mode latent_cluster3 --latent-routing step`。每条 CEM candidate 在每个 imagined step 使用当前预测 latent 做 Z-score + 最近 centroid 路由，再调用对应 predictor；不存在未来 GT 辅助。逻辑回归测试与真实 checkpoint smoke test 均通过。

**长程成功率（goal_offset=50，eval_budget=50；每个 outer 5 eval seeds）**

| eval seed | outer=0 | outer=1 | outer=2 |
|-----------|---------|---------|---------|
| 0 | 60.0% (30/50) | 62.0% (31/50) | 60.0% (30/50) |
| 1 | 60.0% (30/50) | 64.0% (32/50) | 66.0% (33/50) |
| 2 | 62.0% (31/50) | 58.0% (29/50) | 58.0% (29/50) |
| 3 | 56.0% (28/50) | 52.0% (26/50) | 60.0% (30/50) |
| 4 | 58.0% (29/50) | 62.0% (31/50) | 58.0% (29/50) |
| 均值 ± eval-seed std | **59.2 ± 2.3%** | **59.6 ± 4.8%** | **60.4 ± 3.3%** |

**两级汇总与历史协议对照**

| latent 路由协议 | 总均值 | eval-seed SD | outer-seed SD | 相对逐 MPC |
|------------------|-------:|-------------:|--------------:|------------:|
| 逐 MPC（历史） | 59.6% | 4.5% | 1.7% | — |
| **逐预测步（当前）** | **59.7%** | **2.6%** | **0.6%** | **+0.1pp** |

统计口径：总均值为 15 个 outer×eval-seed 结果的均值；eval-seed SD 先在三个 outer 上对同一 eval seed 取均值，再对 5 个 eval seed 求 sample SD；outer-seed SD 对三个 outer 的 5-seed 均值求 sample SD。两种协议复用同一组 baseline eval starts。

**动态切换证据**：15 次正式 run 的 rollout 内 route switch rate 平均 **39.7%**，范围 **33.9%–46.7%**；每个 outer 的三个 cluster 均有实际命中。因此“均值持平”不是因为路由器没有发生切换。

**解释**：逐预测步路由将方法语义修正为真实可部署的动态专家切换，但在当前 TwoRoom / horizon=5 / CEM 配置下，成功率与逐 MPC 路由基本持平（+0.1pp），没有观察到额外平均收益；描述性结果显示 outer 间波动减小。相对 rooms3 50ep 为 +0.5pp，相对 Global-FT 50ep 为 +1.3pp，但均属于边际差异。

**效率 caveat**：本轮三个 outer 与其他用户任务共享 GPU，并行运行时 128 核服务器 load average 一度达到 168；正式 3×5 wall-clock 从 08:42:11 到 09:56:48 UTC。该时间同时包含逐 candidate/逐步路由额外计算和严重资源竞争，**不可作为论文推理延迟**。正式延迟需在独占资源下单独 profile。

输出：

- `results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer{0,1,2}_step_routing_5seed_summary.csv`
- `results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer{0,1,2}_step_routing_5seed_summary.json`
- `results/tworoom_success_rate_latent_kmeanspp_kmeanspp_R50_outer{0,1,2}_step_routing_seed{0,1,2,3,4}/results.json`
- `results/latent_kmeanspp_step_routing_parallel_master.log`

脚本入口：

- `latent_kmeanspp_multirestart.py` — 聚类（已完成）
- `latent_cluster_train_predictors.py` — 按 cluster 微调 predictor（需 `--kmeanspp-label-npz` + `--zscore-params`）
- `tworoom_success_rate_eval.py --mode latent_cluster3 --latent-routing {mpc,step}` — Z-score 路由长程 eval

历史 FAISS 三步命令（对照）：

```bash
bash experiments/tworoom/scripts/run_latent_unsup_cluster_faiss_3seed.sh
CLUSTER_SEED=0 bash experiments/tworoom/scripts/run_latent_cluster_train_predictors_50ep.sh
CLUSTER_SEED=0 bash experiments/tworoom/scripts/run_success_rate_5seed_latent_cluster3_longrange.sh
```

依赖：`faiss-cpu`。**官方协议为 CPU FAISS**（`--device cpu`，默认）；`--device cuda` 可走 PyTorch 全量数据 fit（非官方协议，仅作对照）。

**聚类协议（CPU，909,723 条 192-d 潜向量，K=3，100 iter）**

FAISS `Kmeans` 默认 `max_points_per_centroid=256`，脚本显式保持该默认值。因此 **centroid 拟合最多只用 \(3\times256=768\) 条向量**（子采样），然后对全部 909,723 条做 assign。此前写的 **~0.16 s fit** 是这一子采样协议下的耗时，**不能**表述为“用全部 90 万向量完成 100 轮拟合”。

| 阶段 | CPU (faiss, 官方) |
|------|-------------------|
| 数据加载（5 个 npz 展开+去重，仅一次） | ~19 s |
| 单次 fit（≤768 向量子采样） | **~0.13 s** |
| 单次 assign（全 909,723 向量） | **~0.58 s** |
| 单次 fit + assign | **~0.71 s** |
| 单点分类（assign） | ~0.83 μs/vector |
| 3 seed 聚类（共享加载） | **~35 s 总 wall**（含加载；纯聚类 ~2.1 s） |

瓶颈在 **npz 加载**；纯聚类里 assign 全量向量占主要时间。

**3-seed 聚类稳定性（CPU 官方协议，`faiss_spherical_kmeans_k3_seed{0,1,2}`，对齐 cluster 编号后）**

| seed | cluster0 | cluster1 | cluster2 |
|------|----------|----------|----------|
| 0 | 36.9% | 26.3% | 36.8% |
| 1 | 35.7% | 30.9% | 33.3% |
| 2 | 32.7% | 28.5% | 38.9% |

| 对比 | 标签一致率 | ARI |
|------|-----------|-----|
| seed 0 vs 1 | 48.1% | 0.073 |
| seed 0 vs 2 | 40.7% | 0.030 |
| seed 1 vs 2 | 41.5% | 0.042 |
| 平均 | ~43% | **0.048** |

Centroid 最优对齐后平均余弦相似度：0↔1 **0.59**，0↔2 **0.47**，1↔2 **0.53**。三 seed 标签完全一致 **21.1%**；至少 2/3 seed 一致 **84.5%**。

**结论**：官方 CPU 协议下，3 个 seed 的三分法**差异很大**（ARI ≈ 0.05，接近弱相关）。这既来自 **FAISS 子采样 + 随机初始化** 的不稳定性，也与 K=3 在 192-d 潜空间缺乏唯一自然三分法一致。后续 predictor 微调与长程 eval 需**每个聚类 seed 各跑一套**，最终对 3 个聚类 seed 的结果取 mean±std 汇报。

> 注：曾用 PyTorch GPU 在**全量 909,723 向量**上做 100 轮 fit 得到 ARI ≈ 0.13——那是不同协议，不能与本节 CPU 结果直接对比。

输出目录：

- `results/latent_unsup_cluster/faiss_spherical_kmeans_k3_seed{0,1,2}/`
- `results/tworoom_latent_cluster3_faiss_spherical_kmeans_k3_seed{0,1,2}/`
- `results/tworoom_success_rate_latent_cluster3_exp_main_cluster{0,1,2}_seed{0..4}/`

## 潜向量预处理稳定性筛选（latent_preprocess_stability_v1）

**问题**：在聚类算法、初始化、\(K=3\) 不变时，哪种无监督潜向量预处理能提高不同 clustering seed 之间的划分稳定性？（第一轮不换 K-means++。）

**固定项**：909,723 去重训练潜向量；full-data PyTorch spherical K-means；`niter=100`；随机抽 \(K\) 个样本初始化；clustering seeds `0..19`（20 个）；每种预处理只拟合一次、20 seed 共享；不用 rooms3/priority5 标签、不用 UMAP。

**7 种预处理**：`raw_l2`（V0）、`center_l2`（V1）、`zscore_l2`（V2）、`pca64_l2` / `pca128_l2`（V3–V4）、`pca128_shrink001_l2` / `pca192_shrink001_l2`（V5–V6，\(\alpha=0.01\)）。PCA 对 \(192\times192\) 协方差做确定性 `eigh`。

```bash
bash experiments/tworoom/scripts/run_latent_preprocess_stability.sh
```

脚本：`latent_preprocess_stability.py`（7×20=140 次聚类）。输出：`results/latent_preprocess_stability_k3/`（`per_run_metrics.csv`、`pairwise_stability.csv`、`stability_summary.csv`、`labels/`、`figures/`）。

### 稳定性汇总（v1，固定 100 iter，20 seeds）

| 预处理 | Mean ARI | Median ARI | Consensus \(\bar q\) | \(q\geq0.9\) | 标签一致率 | 100轮仍变化 |
|--------|---------:|-----------:|--------------------:|-------------:|-----------:|------------:|
| raw L2 | 0.158 | 0.137 | 0.645 | 7.0% | 54.2% | 8/20 |
| center L2 | 0.172 | 0.160 | 0.649 | 8.8% | 55.4% | 7/20 |
| **Z-score L2** | **0.207** | **0.185** | **0.676** | **10.8%** | **58.5%** | 7/20 |
| PCA-64 L2 | 0.158 | 0.141 | 0.655 | 10.3% | 54.9% | 4/20 |
| PCA-128 L2 | 0.171 | 0.158 | 0.648 | 8.5% | 55.3% | 7/20 |
| PCA-128 shrink whitening | 0.016 | 0.013 | 0.477 | 0% | 39.2% | 13/20 |
| PCA-192 shrink whitening | 0.017 | 0.013 | 0.486 | 0% | 39.3% | 15/20 |

运行耗时：加载 ~19 s + 140 次聚类 ~19 min（GPU，`fit` ~0.43 s/run）。

### 已验证结论（v1，Verification: ANALYZED）

**Z-score 是目前最好的无监督预处理，但还没有把 K-means 变成稳定算法。**

- Z-score 相对 raw L2：mean ARI **0.158→0.207**（+31%）；Hungarian 标签一致率 **54.2%→58.5%**；median assignment margin **0.140→0.193**；低 margin（\(<0.01\)）样本 **~3.0%→2.24%**。
- 但 mean ARI 仍仅 **0.207**，约 **51%** 样本 consensus \(q<0.7\)。只能算“有所改善”，不能算稳定。
- PCA 降维无明显帮助（PCA-128 ≈ center L2）：被移除的低方差方向不是主要不稳定来源。
- shrink whitening 打散聚类结构（ARI≈0）：重要结构依赖原有方差谱。
- **保留问题**：7–15/20 run 在 100 iter 时 `objective` 仍在变化；低 ARI 不能全部归因于多重局部最优，部分来自未收敛。
- `objective_vs_stability.png` 的 objective **不能跨预处理横向比较**（不同特征空间），只能看同一预处理内 seed 波动。

### 完全收敛复验（v2，`latent_preprocess_convergence_v1`）

仅保留 **raw_l2 / center_l2 / zscore_l2**；K-means 改为 `max_iter=1000`、相对 objective 变化 \(<10^{-7}\) 且连续 10 轮满足才停止；记录标签变化比例；相同 20 seeds。

```bash
bash experiments/tworoom/scripts/run_latent_preprocess_convergence.sh
```

输出：`results/latent_preprocess_convergence_k3/`（脚本 `latent_preprocess_convergence.py`）。

| 预处理 | Mean ARI ± std | Median ARI | Consensus \(\bar q\) | \(q\geq0.9\) | 收敛 (20 seed) | 平均 iter |
|--------|----------------|------------|----------------------|--------------|----------------|----------:|
| **zscore_l2** | **0.226 ± 0.107** | 0.207 | 0.689 | 12.8% | **20/20** | ~136 |
| center_l2 | 0.183 ± 0.106 | 0.172 | 0.661 | 10.5% | 20/20 | ~145 |
| raw_l2 | 0.170 ± 0.099 | 0.149 | 0.646 | 8.2% | 20/20 | ~162 |

完全收敛后 Z-score ARI 仅从 **0.207→0.226**（仍 \(<0.40\)），说明未收敛只解释一小部分差距；**主要瓶颈仍是初始化 / 多重近似等价三分法**。

### K-means++ 多重启（`latent_kmeanspp_multirestart_v1`）

协议：Z-score 预处理；outer seeds 0–19；每 outer seed 内 50 次 K-means++（\(p_i \propto 1-\max_c\cos(x_i,c)\)）；完全收敛；从同一批 inner run 考察 \(R\in\{1,5,10,20,50\}\) 取 objective 最优。

```bash
bash experiments/tworoom/scripts/run_latent_kmeanspp_multirestart.sh
# 若 inner_run_metrics.csv 已存在，仅补汇总：
python experiments/tworoom/latent_kmeanspp_multirestart.py --summary-only --device cuda
```

输出：`results/latent_kmeanspp_multirestart_k3/`（`inner_run_metrics.csv`、`restart_budget_summary.csv`、`zscore_params.npz`、`labels/kmeanspp_R50_outer*.npz`）。

| Inner restarts R | Mean ARI | Median ARI | Consensus \(\bar q\) | \(q\geq0.9\) |
|----------------:|---------:|-----------:|----------------------:|-------------:|
| 1 | 0.168 | 0.155 | 0.658 | 10.8% |
| 5 | 0.253 | 0.228 | 0.722 | 21.0% |
| 10 | 0.348 | 0.339 | 0.784 | 41.4% |
| 20 | 0.385 | 0.375 | 0.812 | 45.8% |
| 50 | **0.483** | **0.488** | **0.856** | **58.7%** |

对照：random init v2（\(R=1\)）Mean ARI **0.226**，Consensus \(\bar q\) **0.689**。

**结论（Verification: SOLID）**：**K-means++ 多重启有效，但 K-means 三分法仍无唯一稳定答案；继续增加 restart 价值有限，应转入下游 predictor 实验。**

- R 从 20→50 仍有提升（0.385→0.483），严格说尚未饱和；但即使 \(R=50\)：Mean ARI 仅 **0.483**，**17.2%** 样本 consensus \(q<0.7\)，不同 outer 划分仍明显不同。
- **近似等价划分已确认**：outer 0 vs 12 的 best objective 差 \(<10^{-5}\)（0.158132 vs 0.158142），但 ARI 仅 **0.291**；全局最高 objective 在 outer 7，与其余 19 个划分的平均 ARI 仅 **0.523**（最低 0.321）。**选最高 objective ≠ 唯一划分**。
- 支持：Z-score 潜空间中 spherical K-means 目标存在**多个质量近似、边界不同的局部最优三分法**。不建议继续跑 100/200 restart 追求唯一划分。

**效率（来自 1000 inner run 原始记录；`restart_budget_summary.csv` 的 `total_cluster_sec=0` 因 `--summary-only` 补汇总，不作正式计时）**

| 项 | 值 |
|----|-----|
| 收敛 | 1000/1000 |
| 单次平均 fit | ~0.693 s |
| \(n\_init=20\) 纯聚类 | ~13.9 s |
| \(n\_init=50\) 纯聚类 | ~34.6 s |
| 加载 + Z-score | ~18.6 s |
| **新数据集 \(n\_init=50\) 离线分区** | **~1 min 以内** |

**论文主配置（聚类阶段）**

```text
zscore_l2
spherical K-means++  (p_i ∝ 1 - max_c cos(x_i, c))
K = 3, n_init = 50, max_iter = 1000
rel_tol = 1e-7, patience = 10
select = maximum objective
```

\(n\_init=20\) 可作快速版或消融。Rollout 路由须用 `zscore_params.npz` 的 \(\mu,\sigma,\varepsilon\)：\(x_t=\mathrm{normalize}((z_t-\mu)/(\sigma+\varepsilon))\)，再与 centroid 比余弦。

### 轻量 landmark 谱分割（`latent_landmark_spectral_v2`，2026-07-16）

目的：保留“仅用潜向量自动划分 + 多 predictor”的当前方法接口，同时解决 spherical K-means++ 的多解不稳定问题，且离线划分开销不显著高于现有版本。

固定流程：

```text
909,723 个去重训练潜向量
→ train-set Z-score + L2 normalize
→ 按 episode 覆盖抽取 20,000 landmarks
→ GPU 分块 exact cosine 30-NN
→ self-tuned 对称 kNN 图
→ normalized-adjacency eigsh + 3-way spectral K-means
→ 每个谱簇拟合 16 个 spherical prototypes
→ nearest prototype → owner cluster 给全量数据分配标签并在线路由
```

算法本身不依赖 TwoRoom 几何标签，但当前数据加载器复用了 TwoRoom 的 priority5 embedding caches，因此 CLI 暂时只允许 `--dataset tworoom`。迁移到 PushT 等数据集时需要先补统一的 cache manifest/loader，不能直接复用当前默认路径。

当前论文主协议仍固定 `K=3`；`AUTO_K=1` 只是可选诊断，根据预先声明的 normalized-Laplacian 最大 eigengap 在 `[KMIN,KMAX]` 内自动选簇数，不会静默改变现有主实验。

实现文件：

- `latent_landmark_spectral.py`
- `scripts/run_latent_landmark_spectral.sh`
- `scripts/run_latent_spectral_train_predictors_50ep.sh`
- `scripts/run_success_rate_5seed_latent_spectral_longrange.sh`
- `scripts/run_latent_spectral_short_horizon.sh`
- `test_latent_landmark_spectral.py`
- `tests/test_spectral_manifest_contract.py`
- `tests/test_latent_cluster_step_routing.py`
- `tests/test_trajectory_switch_rollout.py`

运行命令：

```bash
# 单 seed 快速产生一个可训练 artifact
GPU=0 SEEDS=0 bash experiments/tworoom/scripts/run_latent_landmark_spectral.sh

# 正式稳定性检查（共享一次数据加载）
GPU=0 SEEDS=0,1,2 bash experiments/tworoom/scripts/run_latent_landmark_spectral.sh

# 可选 auto-K 诊断；不是当前 TwoRoom 主配置
GPU=0 SEEDS=0 AUTO_K=1 AUTO_K_MIN=2 AUTO_K_MAX=6 \
  bash experiments/tworoom/scripts/run_latent_landmark_spectral.sh

# 下游 predictor（尚未启动）
GPU=0 SPECTRAL_SEED=0 TRAIN_SEED=42 \
  bash experiments/tworoom/scripts/run_latent_spectral_train_predictors_50ep.sh

# 长程评测；LATENT_ROUTING 可设为 mpc 或 step
GPU=0 SPECTRAL_SEED=0 TRAIN_SEED=42 LATENT_ROUTING=mpc \
  bash experiments/tworoom/scripts/run_success_rate_5seed_latent_spectral_longrange.sh

# 同一 artifact/predictor 的短程 rollout；LATENT_ROUTING 可设为 mpc/step/oracle_gt_step
GPU=0 SPECTRAL_SEED=0 TRAIN_SEED=42 LATENT_ROUTING=mpc \
  bash experiments/tworoom/scripts/run_latent_spectral_short_horizon.sh

# K-means++ 保持部署匹配的 transition-start anchor（offset=0）
GPU=0 OUTER_SEED=0 ROUTE_LABEL_OFFSET_STEPS=0 \
  bash experiments/tworoom/scripts/run_latent_kmeanspp_train_predictors_50ep.sh
```

固定 K 输出目录使用配置指纹：`results/latent_landmark_spectral_k<K>/spectral_cfg<fingerprint>_K<K>_M20000_k30_P<P>_seed<seed>/`；auto-K 单独写入 `results/latent_landmark_spectral_auto_k/`，不会改写 K=3 的 latest summary。旧的 `spectral_M20000_k30_P16_seed*` 仍可读取。`stability_summary.json` 是带 `artifacts_by_seed` 的 latest alias，同时保留不可覆盖的 `stability_summary_cfg<fingerprint>.json`。下游脚本优先从 summary 解析 artifact，也允许显式设置 `SPECTRAL_ROOT` / `ARTIFACT_DIR`；无 summary 时只有唯一候选才会继续，不能再静默硬编码 P16。predictor、长程和短程默认输出目录都包含 artifact basename，防止不同谱配置互相覆盖。每个 artifact 包含全量 `cluster_labels.npz`、routing prototypes/owners、Z-score 参数、landmark 诊断和 `cluster_meta.json`。

#### 已完成的 3-seed 划分结果

| Landmark seed | 端到端耗时（含共享 load） | 纯分区耗时 | Holdout Macro-F1 | 全量 cluster fractions |
|---:|---:|---:|---:|---|
| 0 | 20.19 s | 5.79 s | 99.29% | 36.21% / 32.40% / 31.40% |
| 1 | 19.26 s | 4.86 s | 98.94% | 32.44% / 35.96% / 31.60% |
| 2 | 19.04 s | 4.64 s | 99.16% | 35.81% / 33.01% / 31.18% |

三 seed 共用一次 14.40 s 加载后，总 wall time 为 **32.84 s**。三个图均为单连通分量；seed 0 的 GPU exact kNN 约 0.14 s、eigensolver 约 0.50 s，prototype holdout 每类 recall 最低 98.38%。

| Pair | Full-label ARI | NMI |
|---|---:|---:|
| seed 0 vs 1 | 0.9613 | 0.9342 |
| seed 0 vs 2 | 0.9600 | 0.9352 |
| seed 1 vs 2 | 0.9589 | 0.9341 |
| **Mean** | **0.9601** | **0.9345** |

对照 K-means++ `R=50` 的 20-seed mean ARI 为 **0.483**。按预注册的描述性阈值（`<0.433` 退步，`≥0.50` 至少改善，`≥0.60` 明显改善），landmark spectral 在 TwoRoom 上属于**明显更稳定**。这只证明划分稳定性与可部署压缩已通过，尚未证明下游控制优于 K-means++；predictor 尚未启动。

谱簇不是人工 rooms3 的复刻：三 seed 与 rooms3 对齐后的 ARI 约 **0.429–0.435**、最优标签对齐 accuracy 约 **69.5–69.9%**；它比 K-means++（ARI 约 **0.008–0.025**）更贴近全局空间结构，但 central cluster 同时覆盖 doorway 及其邻域。因此论文中应称为无监督潜空间分区，不能称为恢复了人工 rooms3。

#### Artifact / 路由一致性（schema v2）

- 20k 谱标签仅用于产生 pseudo-label；训练和推理使用同一套 prototype-owner rule；当前 K=3、P=16 主配置共有 48 个 routing prototypes，其他 K/P 不再假设固定为 48。
- 全量 909,723 个 timestep 均有最终部署标签；随机抽查 10,000 点，离线 artifact label 与 prototype router 一致率 **100%**。
- seed 0 用相同配置独立重跑，full-label SHA-256 完全一致（`0201553a...24fa3`），确定性复验通过。
- 旧 artifact 自动回退为 `centroids + identity owner`，现有 K-means++ 推理不变。新 artifact 额外记录代码、输入 cache、语义数组、环境版本和所有输出数组的 SHA-256；加载、续训和评测均 fail closed。
- predictor manifest 绑定 cluster artifact、Z-score、base checkpoint、train starts、合并 embedding cache 和每个输出 checkpoint 的 SHA-256；`ONLY_CLUSTERS` 只允许在 immutable fingerprint 完全一致时恢复，并通过 staging + 锁内原子 merge 避免并发丢记录。完整重训若目标 manifest 已存在会在训练前失败，只有显式 `OVERWRITE_EXISTING=1` 且 immutable fingerprint 一致才允许替换。
- artifact 生成、predictor 提交和 manifest 写入均有中断恢复：普通异常和 Ctrl+C 都会清除临时目录并恢复旧版本。训练端对 artifact 做读取前后双 SHA-256 快照；评测端让同一把 manifest 锁覆盖“哈希验证 + checkpoint 加载”，避免并发覆盖导致 TOCTOU。评测结果还嵌入当时 manifest、artifact 和 predictor 的哈希快照，后续原地恢复不会抹去旧结果的证据链。
- 新谱 artifact 采用 `route_anchor=transition_start`。当前 TwoRoom 在线评测没有堆叠三帧 observation history：`info["pixels"]` 与 `info["proprio"]` 的时间轴长度均为 1；CEM 只新增 sample 轴，因此索引 0 就是当前 MPC rollout 起点。predictor 训练应按 `train_start+0` 的标签选择样本。`history_size=3` 是 predictor 的最大自回归上下文长度，不代表在线输入包含三帧真实观测。
- MPC 推理不再通过全局替换一个 predictor 完成路由；它会为 batch 中每个环境分别选择专家，扩展到该环境的全部 CEM candidates，并在一次 imagined rollout 内固定；同一次 MPC 的重复 CEM rollout 调用复用该路由。逐步模式第一次从 rollout 起点 latent 路由，之后只根据上一时刻预测 latent 动态切换。
- **公平性结论**：历史 K-means++ predictor 使用 `train_start+0`，这与当前在线 MPC 起点路由一致。谱方法也必须使用相同的 `transition_start / offset=0` 协议；不需要为所谓 `history_end / +10` 公平性重训 K-means++。
- 现有 `select_best_by_eval` 名称容易误解：实现监控的是同一 cluster 训练 embedding pool 上的 predictor loss，因此它是 training-objective checkpoint selection，不是 held-out validation。成功率 test set 仍保持独立；论文中不应把该字段称为 validation selection。

#### v2 修复验证（2026-07-16）

- Python/shell 语法检查通过；谱算法 **8** 个测试、manifest 与中断事务 **6** 个测试、在线路由 **4** 个测试、离线路由 **7** 个测试全部通过。
- 三个旧 K=3 artifact 均通过全量训练映射 dry-run：每个覆盖 **693,728** 条 train transitions，`route_label_offset_steps=0`，无缺失标签。
- seed 0 用当前实现完整重放；新旧 `global_idx`、full labels、centroids、routing prototypes、prototype owners、Z-score mean/std 均逐数组 **exact equal**。因此 v2 的 provenance、原子写入、auto-K 和通用 K/P 支持没有改变固定 K=3 主协议的数值结果。
- 使用现有真实 K-means predictor checkpoint 完成 manifest 锁内加载 smoke test，成功构造 `LatentClusterSwitchJEPA`。旧 v1 manifest 仍可兼容加载；新谱 predictor 一旦训练会强制使用 schema v2 的完整哈希契约。

#### Spectral K3-50 公平对照（运行中，2026-07-16）

目标：只替换无监督分区算法，在其余条件完全一致时，将 landmark spectral 与现有 K-means++ R50 做主对照。

| 控制项 | K-means++ R50 | Landmark spectral |
|---|---|---|
| partition seeds | outer 0/1/2 | landmark 0/1/2 |
| 分区数 | K=3 | K=3 |
| train transitions | 693,728 | 693,728 |
| route anchor | transition start，offset=0 | transition start，offset=0 |
| predictor epochs | 50 | 50 |
| predictor train seed | 42 | 42 |
| base checkpoint | official `lewm_object.ckpt` | 相同 |
| optimizer recipe | `lr=5e-5, batch=128, weight_decay=1e-3` | 相同 |
| 主评测路由 | per-MPC | per-MPC |
| eval protocol | 原 5 个 eval seeds 与起点 | 完全复用 |

Spectral predictor 输出：

- `results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed0_trainseed42/`
- `results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed1_trainseed42/`
- `results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed2_trainseed42/`

三个 predictor 训练任务于 2026-07-16 13:17 UTC 启动，分别使用 GPU 0/1/2；每个任务限制 4 个 CPU 线程并以 `nice=10` 运行。训练完成后使用 `run_success_rate_5seed_latent_spectral_longrange.sh` 做逐 MPC 的 3 partition-seed × 5 eval-seed 长程评测，再按 partition seed 汇总 mean ± SD；不会根据 test 成功率选择某个 spectral seed。

为估计 predictor 微调随机性，补跑 train seeds `0/625`，每个 train seed 都覆盖相同的三个 spectral partition seeds。train seed 0 于 2026-07-16 13:30 UTC 在 GPU 3/4/6 启动；train seed 625 排队等待 train seed 42 释放 GPU 0/1/2，随后自动启动。全部实验复用同一份已冻结 encoder 的 `P_train_global_merged_embeddings.npz`，不重新编码；六个新输出目录通过硬链接复用同一 inode 的 1.27 GB 缓存，避免重复占用约 7.6 GB。为避免多个进程同时解压读取该缓存，train seed 625 使用串行加载屏障：只有前一个 partition 出现 `.cluster0.training.lock`、确认进入训练阶段后，才启动下一个 partition；完成加载后三个 GPU 仍并行训练。每个进程均限制 4 个 CPU 线程并以 `nice=10` 运行。

新增调度入口：

- `scripts/run_latent_spectral_trainseed625_after_seed42.sh`：等待 seed 42 的三个 predictor manifest；对 seed 625 执行串行缓存加载、三 GPU 并行训练；全部训练结束后依次评测 train seeds 42/625。
- `scripts/run_latent_spectral_seed_eval_after_train.sh`：适用于任意 predictor train seed；等待三个 predictor manifest 后运行 3 partition-seed × 5 eval-seed 的逐 MPC 长程评测，并与相同 train seed 的 K-means++ R50 结果汇总比较。三个评测进程错峰 20 秒启动。
- train seed 0 使用同一通用评测入口，在训练完成后于 GPU 3/4/6 自动评测；所有失败均停止并记录，不自动重试。

#### Inference-tensor 路由缓存修复与评测重启（2026-07-17）

九套 Spectral K3-50 predictor（3 partition seeds × train seeds 0/42/625）均已训练完成。首次自动长程评测在 eval seed 0 的第一次 MPC rollout 统一失败：`LatentClusterSwitchJEPA._tensor_identity` 读取 `tensor._version`，但 `torch.inference_mode` 创建的 tensor 不跟踪 version counter。该失败发生在控制评测开始阶段，不影响已有 partition artifact 或 predictor checkpoint。

修复保持普通 tensor 的 version counter；仅对 inference tensor 使用 `version=None`，并继续以 device、dtype、storage pointer、offset、shape 和 stride 构造缓存键。MPC 缓存同时保留上一 observation storage 的强引用，因此新 observation 的 allocator pointer 不会与旧缓存碰撞；一次 CEM solve 内的 expanded observation 保持只读，仍可安全复用同一路由。

验证：

- `test_latent_cluster_step_routing.py`：5/5 通过，新增 inference tensor identity 回归测试；
- 真实 spectral seed 0 / train seed 42 checkpoint 的 1-episode smoke test：退出码 0，成功完成控制评测；路由缓存 `2 misses / 58 hits`；
- 正式重启入口：`scripts/run_latent_spectral_all_trainseed_eval_rerun.sh`。先在 GPU 0/1/2 与 3/4/6 并行评测 train seeds 42/0，成功后再在 GPU 0/1/2 评测 train seed 625；每个进程限制 4 个 CPU 线程，单组硬超时 2 小时，失败不再自动重试。

首次重启中，train seeds 0/42 的 30 次控制评测均成功完成；seed 0 仅在后处理汇总阶段因历史 K-means++ CSV 使用 `eval_seed` 而新 CSV 使用 `seed` 触发 `KeyError`。`aggregate_partition_seed_success.py` 已兼容两种 schema，直接复用完成的控制结果，没有重跑；随后 seed 625 的 15 次评测全部成功。最终共 **45/45** 个结果文件完整生成，未再出现 inference-tensor 异常。

**Spectral K3-50 最终长程成功率（每格先对 5 个固定 eval seeds 取均值）**：

| predictor train seed | spectral partition 0 | partition 1 | partition 2 | Spectral mean | K-means++ R50 mean | paired Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 62.4% | 59.2% | 63.2% | **61.60%** | 58.27% | **+3.33pp** |
| 42 | 61.2% | 61.6% | 62.8% | **61.87%** | 59.60% | **+2.27pp** |
| 625 | 61.2% | 62.0% | 58.8% | **60.67%** | 57.47% | **+3.20pp** |
| train-seed mean ± sample SD | — | — | — | **61.38 ± 0.63%** | **58.44 ± 1.08%** | **+2.93 ± 0.58pp** |

`±` 的统计单位是三个 predictor fine-tuning seeds；每个 train-seed 点内部已对三个 partition seeds 与五个固定 eval seeds 求均值。45 个结果不能作为 45 个独立重复。三个 train seeds 上 Spectral 均优于对应 K-means++，但 `n=3` 只支持描述性稳定性结论，不作显著性检验。

### 谱方法消融：分区策略（K=3，50 epochs）

该消融只改变 predictor 的分区策略，用于回答三个问题：空间多 predictor 是否优于不分区的单 predictor、谱流形分区是否优于直接 K-means++、以及自动谱分区能否达到人工 K=3 几何划分的水平。

统一控制项：冻结相同的官方 encoder；使用相同的 693,728 条 train transitions、episode split、base checkpoint、optimizer recipe 和 50 epochs；predictor train seeds 固定为 `0/42/625`；复用相同的五个 eval seeds、任务起点与 per-MPC 路由协议。三个分区方法均为 K=3。Global-FT50 与分区方法的训练 transition-updates 对齐，但单 predictor 与三 predictor 的参数存储量并不相同，因此这里的 `compute-matched` 特指训练样本更新量，不代表参数量或模型存储完全相同。

| 方法 | 分区方式 | Predictor 数 | 长程成功率（train-seed mean ± sample SD） |
|---|---|---:|---:|
| Global-FT50 | 无分区 | 1 | 58.13 ± 1.22% |
| K-means++ K3-50 | 自动欧氏/球面聚类 | 3 | 58.44 ± 1.08% |
| rooms3-50 | 人工几何 K=3 | 3 | 59.87 ± 1.51% |
| **Spectral K3-50** | **自动流形谱分区** | **3** | **61.38 ± 0.63%** |

| 配对比较 | 三个 train seeds 上的 mean Δ ± sample SD |
|---|---:|
| Spectral − Global-FT50 | **+3.24 ± 0.76pp** |
| Spectral − K-means++ K3-50 | **+2.93 ± 0.58pp** |
| Spectral − rooms3-50 | **+1.51 ± 2.12pp** |

结论边界：Spectral 在三个 train seeds 上均高于对应的 Global-FT50 与 K-means++，支持“空间特化有效”以及“流形感知分区优于直接聚类”这两个描述性结论。相对 rooms3-50，Spectral 的总体均值更高，但 seed 625 上低 0.93pp，因此应表述为“自动谱分区达到人工 K=3 划分水平并在均值上更高”，不能声称稳定支配人工划分。统计单位只有三个 predictor fine-tuning seeds，`n=3` 不用于显著性检验。

#### 补充强基线：Joint-Continue 1ep 与 Random-K3（2026-07-17）

为进一步拆分“冻结 encoder 的 predictor 后训练”“多 predictor 容量”与“有结构的潜空间分区”三种因素，新增以下两个 TwoRoom 对照；继续沿用现有 LeWM 标准评测、固定 5 个 eval seeds 和完全相同的短/长程起点，不另建 final-test 协议。

1. **Joint-Continue 1ep**：从官方 `lewm_object.ckpt` 出发，对 encoder、projector、action encoder、predictor 与 pred-proj 做一次完整训练数据遍历；目标函数保持官方 `prediction loss + 0.09 * SIGReg`，数据窗口、90/10 split、batch size 128、学习率 `5e-5`、weight decay `1e-3` 与官方配置一致。官方发布物只有模型参数，没有 optimizer/scheduler state，因此该对照使用 fresh AdamW、无 scheduler；论文中必须写成“released-checkpoint joint continuation with a fresh optimizer”，不能声称精确恢复原训练 optimizer trajectory，也暂不称为 compute-matched。
2. **Random-K3 / Random-Voronoi K3-50**：复用 K-means++ 的训练集 Z-score + L2 预处理；每个 partition seed 从训练潜向量均匀无放回抽取 3 个原型，不做 K-means、restart 或目标选择，再用 maximum-cosine Voronoi rule 给所有 transition 分区。这样测试时仍可只根据当前 latent 路由；K=3、3 个 predictor、50 epochs、总 transition-updates、base checkpoint、optimizer recipe 与 K-means++/Spectral 完全一致。它是无学习分区结构的多 predictor 容量控制，而不是无法对新状态路由的独立随机 transition 标签。

**精度与硬件对齐（canonical policy）**：Spectral 的 latent cache、谱分区输入和 K3-50 predictor 微调均为 FP32，因此 Joint-Continue 的 canonical 对照也固定为 FP32。`joint_continue_tworoom.py` 新增 `--precision {fp32,bf16-mixed}` 且默认 `fp32`，`scripts/run_joint_continue_tworoom_1ep.sh` 显式传 `--precision fp32`；manifest 必须记录 `config.precision=fp32`。两者均使用单张 RTX 4090、batch size 128、`OMP/MKL/OPENBLAS_NUM_THREADS=4` 和 `nice=10`；Joint 的原始图像 DataLoader 使用 4 workers。此前的 BF16 Joint checkpoint、训练日志和长短程结果已移至 `results/archive/joint_continue_bf16_trainseed42_20260717/`，不得进入主表或 FP32 耗时比较；FP32 checkpoint 完成后须重新跑完全相同的 5-seed short/long 评测并覆盖 canonical 汇总。

三个 Random-Voronoi artifacts 已生成并通过 schema-v2 artifact/manifest dry-run，693,728 个训练 transition 全部覆盖：

| partition seed | cluster0 | cluster1 | cluster2 |
|---:|---:|---:|---:|
| 0 | 202,975 | 275,130 | 215,623 |
| 1 | 225,377 | 232,431 | 235,920 |
| 2 | 231,676 | 282,582 | 179,470 |

正式批次固定 predictor train seed 42：Random-K3 使用三个预先声明的 partition seeds，各训练一套 K3-50 predictor，并完成 short (`goal_offset=25`) 与 long (`goal_offset=50`) 的相同 5-seed 配对评测。三个 predictor manifest 与六份 5-seed summary 均已生成。Random-K3 三个进程通过全局 cache-load lock 串行读取共享的 1.27 GB merged embedding cache，进入训练后才并行，避免同时解压读取。

**Random-Voronoi K3-50 成功率（mean ± sample SD across five eval seeds）**：

| partition seed | long | short |
|---:|---:|---:|
| 0 | 60.8 ± 1.10% | 90.8 ± 2.28% |
| 1 | 58.8 ± 3.35% | 90.4 ± 2.97% |
| 2 | 57.6 ± 3.29% | 92.0 ± 4.24% |

以每个 partition seed 的 5-eval-seed 均值为一个统计单位，再对三个 partition seeds 汇总：long 为 **59.07 ± 1.62%**，short 为 **91.07 ± 0.83%**（sample SD across partition seeds）。Random-Voronoi 在没有学习分区结构的情况下也能取得明显高于原始 baseline 的控制成功率，说明多 predictor 容量与局部后训练本身具有贡献；但 long 的 partition-seed 波动高于 short，且该对照不能替代 Spectral/K-means++ 的结构化分区比较。

Joint-Continue 的 canonical FP32 训练已在相同硬件限制下重新启动；原 BF16 结果仅归档，不与 Spectral 的 FP32 结果混用。

新增入口：

- `latent_random_voronoi.py`
- `joint_continue_tworoom.py`
- `scripts/run_random_voronoi_k3.sh`
- `scripts/run_random_voronoi_train_predictors_50ep.sh`
- `scripts/run_random_voronoi_eval.sh`
- `scripts/run_joint_continue_tworoom_1ep.sh`
- `scripts/run_joint_continue_tworoom_eval.sh`
- `scripts/queue_tworoom_control_baselines.sh`

队列状态：`results/tworoom_control_baselines_queue/queue.log`。CPU smoke test 已验证 Joint-Continue 的官方数据变换、联合反传、checkpoint 原子保存和 manifest；该 smoke 使用 `max_batches=1`、`sigreg_num_proj=8` 且标记 `formal_full_epoch=false`，不得纳入论文结果。正式结果完成后再补 predictor train seeds `0/625`，并按现有约定先对每个 train seed 的 5 个 eval seeds 求均值，再以 train seed 为统计单位报告 mean ± sample SD。

当前最佳人工配置 priority5-30 **不纳入本消融**：它同时使用 K=5 与 30 epochs，和上述 K=3、50 epochs 协议不对齐，直接差值不能归因于谱分区。priority5-30 继续保留在人工划分/微调轮次搜索结果中；若后续需要正面对比，应另设对齐配置的 K/epoch 敏感性实验。

自动评测与公平汇总由以下两个新增入口完成：

- `scripts/run_latent_spectral_k3_50_fair_eval_after_train.sh`：最多等待 4 小时；只有三个 schema-v2 predictor manifest 全部原子提交后，才在 GPU 0/1/2 并行启动三个 partition seed 的 5-seed 长程评测。若训练进程提前退出或评测失败，只记录错误，不自动重试。
- `aggregate_partition_seed_success.py`：同时读取 Spectral 3×5 与现有 K-means++ R50 3×5 summary；报告每个 partition 的 5-seed 均值、partition-seed SD、15-run grand mean，以及先对三个 partition 取均值后按相同 eval seed 配对的差值。该差值只作描述性比较，不作为五个 seed 的显著性检验。

最终比较输出：

- `results/tworoom_success_rate_latent_spectral_k3_50_vs_kmeanspp_R50_trainseed42_mpc.json`
- `results/tworoom_success_rate_latent_spectral_k3_50_vs_kmeanspp_R50_trainseed42_mpc.csv`

### 主实验结果：潜向量聚类 → per-partition predictor（2026-07-13/14）

Step 1–3、逐预测步动态路由补测与 Global-FT 50ep 强 baseline（训+eval）均已完成（见上方长程成功率表）。下表以当前预期的逐预测步协议为准；历史逐 MPC 结果保留为对照。

| 可能结果 | 含义 | 本次 |
|----------|------|------|
| 三种划分成功率均提升且相近 | 局部化有效，不要求唯一聚类边界（理想） | **是**（59.7%，outer-seed SD 0.6pp，vs baseline +10.5pp） |
| 成功率随 outer seed 剧烈变化 | K-means 分区不够可靠，应换聚类目标而非加 restart | **否**（outer 均值 59.2–60.4%） |
| cluster-switch 优于 geometry rooms3 50ep | 无监督 3-cluster 路由不低于有监督 3-region 路由 | **边际**（59.7% vs 59.2%，+0.5pp） |
| cluster-switch 优于同轮次 Global-FT | 分 cluster 特化 + 潜空间路由有额外收益 | **边际（成功率 +1.3pp；rollout MSE step10 +0.003 vs Global-FT，见上方四模型表）** |
| 逐预测步优于逐 MPC latent 路由 | rollout 内动态切换带来额外控制收益 | **否（59.7% vs 59.6%，+0.1pp，描述性持平）** |

注：190 个 pairwise ARI 非独立样本，误差棒仅作描述性波动。

## 已完成结果：geometry rooms3 潜向量线性可区分性（episode split + timestep dedup）

数据：P_train 缓存展开 2,774,912 条 → rooms3 过滤后按 global timestep 去重为 **909,723** 条唯一潜向量（去掉 1,865,189 条窗口重叠重复）。episode 划分 7000 / 1500 / 1500，对应 **637,122 / 136,168 / 136,433** 条向量。类别比例：left ~47%，doorway ~6%，right ~47%。

### 总表（test episode 潜向量）

| 探针 | accuracy | balanced acc | macro-F1 | left F1 | doorway F1 | right F1 |
|------|----------|--------------|----------|---------|------------|----------|
| Linear Softmax Probe | 99.88% | 99.72% | **99.60%** | 99.93% | 98.92% | 99.95% |
| RFF-RBF Probe | 99.94% | 99.65% | **99.78%** | 99.97% | 99.41% | 99.96% |

### Val episode（一致性检查）

| 探针 | accuracy | balanced acc | macro-F1 | doorway F1 |
|------|----------|--------------|----------|------------|
| Linear Softmax Probe | 99.89% | 99.76% | 99.65% | 99.07% |
| RFF-RBF Probe | 99.94% | 99.72% | 99.81% | 99.48% |

### 结论

1. **rooms3 is almost perfectly linearly decodable from the LeWM latent representation**：去重后 Linear Softmax Probe test macro-F1 **99.60%**（此前未去重因滑动窗口重复膨胀至 99.85%）。
2. **线性边界已足够**：RFF-RBF Probe test macro-F1 99.78%，仅高 ~0.2pp。
3. **doorway 相对最弱**（test F1 ~98.9–99.4%），与其样本占比最低（~6%）及边界过渡带一致。
4. episode-level holdout + timestep 去重下仍近满分，支持基于潜空间做自动区域划分。

#### 5-seed 稳定性检验（seed = 0,1,2,3,4）

协议与 priority5 相同：每个 seed 联合控制 episode 划分、Linear Probe 初始化/minibatch、RFF 随机特征；同一 seed 下两种探针共享 episode split。

```bash
bash experiments/tworoom/scripts/run_geometry_latent_probe_rooms3_multiseed.sh
```

日志：`results/geometry_latent_probe_rooms3_multiseed/run_multiseed.log`（5 seeds × 2 probes，~236s）

**Test episode 汇总（mean ± std，5 seeds）**

| 探针 | accuracy | balanced acc | macro-F1 | left F1 | doorway F1 | right F1 |
|------|----------|--------------|----------|---------|------------|----------|
| Linear Softmax Probe | 99.90% ± 0.01% | 99.71% ± 0.04% | **99.67% ± 0.04%** | 99.93% ± 0.01% | 99.12% ± 0.11% | 99.96% ± 0.01% |
| RFF-RBF Probe | 99.91% ± 0.01% | 99.56% ± 0.06% | **99.72% ± 0.03%** | 99.95% ± 0.01% | 99.24% ± 0.08% | 99.96% ± 0.00% |

**配对差值** \(\Delta_s = \mathrm{MacroF1}_{\mathrm{RFF},s} - \mathrm{MacroF1}_{\mathrm{Linear},s}\)：

| 统计量 | 值 |
|--------|-----|
| mean ± std | **+0.05pp ± 0.02pp** |
| per-seed | +0.06, +0.07, +0.03, +0.04, +0.04 pp |
| RFF > Linear | **5 / 5 seeds** |

5-seed 均值 +0.05pp 远小于 priority5 的 +0.78pp；RFF 虽在全部 seed 上略高，但增益极小且 std 仅 0.02pp，**rooms3 几乎完全线性可分，非线性边界无实质必要**。

输出文件（JSON 中模型键：`linear_softmax_probe`、`rff_rbf_probe`）：

- `experiments/tworoom/results/geometry_latent_svm_rooms3/geometry_latent_svm_rooms3.json`
- `experiments/tworoom/results/geometry_latent_svm_rooms3/geometry_latent_svm_rooms3_metrics.csv`
- `experiments/tworoom/results/geometry_latent_svm_rooms3/geometry_latent_svm_rooms3_split.npz`
- `experiments/tworoom/results/geometry_latent_probe_rooms3_multiseed/geometry_latent_probe_rooms3_multiseed.json`
- `experiments/tworoom/results/geometry_latent_probe_rooms3_multiseed/geometry_latent_probe_rooms3_multiseed_summary.csv`

## 冻结 Encoder 的快速重编码后端（2026-07-17）

`trajectory.py` 的 embedding precompute 默认由逐 batch、逐 start 的同步 HDF5 读取，改为 **exact-start DataLoader** 流水线。这里只替换 I/O 后端，不采用官方 `HDF5Dataset` 的窗口集合与 split，因此保持本实验已有协议不变：

- 严格消费原有 `train_global_reference_starts.npy` 或 region starts，顺序不变；
- `history_size=3`、`num_preds=1`、`frameskip=5` 与像素预处理不变；
- action block 的展开顺序、train-global 归一化和末尾零 padding 不变；
- 输出仍为 `P_*_embeddings.npz`，键为 `emb`、`act_emb`、`region_starts`；
- HDF5 句柄由每个 worker 延迟独立打开，action 列在内存缓存；默认 4 workers、prefetch factor 2，不额外扩大 CPU 占用。

新增参数：

```bash
--embedding-loader dataloader \
--encode-workers 4 \
--encode-prefetch-factor 2
```

`dataloader` 已设为默认；如需复核历史路径，可显式传 `--embedding-loader legacy`。已有正确缓存仍优先复用，只有缺少缓存、cache starts 不匹配或显式 `--force-reencode` 时才会启动重编码。

一致性与计时命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 \
.venv/bin/python experiments/tworoom/validate_embedding_loader.py \
  --sample-size 1024 --batch-size 128 --workers 4 --device cuda
```

固定抽样 1,024 条 transition 的结果：pixels、raw action blocks、vectorized action blocks、starts 均逐元素相同；`emb` 与 `act_emb` 均为 float32 且逐元素完全相同，最大绝对误差为 0。相同 checkpoint 与 batch size 下，旧后端 61.56 s，新后端 16.41 s，实测加速 **3.75×**。该计时只用于后端对比，不从稀疏抽样结果外推完整数据集耗时。验证产物：

- `experiments/tworoom/results/embedding_loader_validation/validation.json`

### 推荐：唯一 timestep 无损重编码

后续需要重新生成 latent cache 时，**推荐优先使用**独立脚本
`experiments/tworoom/unique_timestep_reencode.py`。相邻 transition windows
共享大量图像帧；该脚本先构造所有窗口的 global frame indices，对唯一 timestep
只运行一次 frozen encoder，再通过 inverse index 重建原来的 `(N, 4, 192)` 窗口。
它同时保持原有 transition 顺序、train-global action normalization、action padding、
FP32 精度以及 `emb`、`act_emb`、`region_starts` 三个输出键不变。

为保证**逐位一致**而不只是数值接近，脚本还保留原始 visual batch shape。TwoRoom
全局训练集仅有 15 个 timestep 横跨完整批和末尾残批，需要按两种 batch shape
各编码一次；因此 909,723 个唯一 timestep 对应 909,738 个 exact shape keys，补齐
CUDA 尾批后实际编码 910,208 帧，仍远少于原始 2,774,912 个 frame slots。

推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 nice -n 10 \
.venv/bin/python experiments/tworoom/unique_timestep_reencode.py \
  --starts experiments/tworoom/results/tworoom_geometry_train_region_predictors/train_global_reference_starts.npy \
  --action-norm-starts experiments/tworoom/results/tworoom_geometry_train_region_predictors/train_global_reference_starts.npy \
  --output experiments/tworoom/results/unique_timestep_reencode/P_train_global_embeddings.npz \
  --report experiments/tworoom/results/unique_timestep_reencode/P_train_global_embeddings.report.json \
  --num-workers 4 --cpu-threads 4 --device cuda
```

**TwoRoom 全局训练集耗时口径**（693,728 transitions，RTX 4090，FP32，4 workers）：

- 核心重编码预计约 **21.6–22.1 分钟**；
- 加上 checkpoint 加载、action normalization、inverse reconstruction 和压缩 NPZ
  写盘后，端到端总时间预计约 **23–24 分钟**，实际排期建议按约 25 分钟预留；
- 同硬件、同 8,192-window 前缀上，当前 DataLoader 后端为 120.10 s，唯一 timestep
  后端两次为 15.28–15.66 s，核心实测加速约 **7.7–7.9×**；
- 单看视觉 kernel 工作量，实际从 2,774,912 降至 910,208 帧，约减少
  **67.2%**，对应约 **3.05×**；总加速更高是因为按 HDF5 chunk 顺序读取也消除了
  大部分重复解压与 I/O 等待。

严格一致性已经在完整 doorway cache（38,552 transitions，154,208 个 latent
positions）上验证：`emb`、`act_emb`、`region_starts` 均逐元素相同，mismatch count
为 0，最大绝对误差为 0；原始尾批只有 3 个 transitions 的边界情况也单独通过。
这里的“输出一致”指 NPZ 内数组、shape、dtype 与顺序完全一致；压缩 NPZ 容器本身
可能因为 ZIP metadata 不同而具有不同文件 SHA-256。验证和计时记录：

- `experiments/tworoom/results/unique_timestep_reencode_validation/doorway_full_exact.report.json`
- `experiments/tworoom/results/unique_timestep_reencode_validation/global_first8192_benchmark_v2.report.json`
- `experiments/tworoom/results/unique_timestep_reencode_validation/original_dataloader_first8192_benchmark.json`

已有正确 cache 不需要重编码，基于旧 cache 完成的 Spectral/K-means++ 划分、predictor
训练与控制评测也不需要重跑；新脚本用于后续新增数据集、缓存缺失或主动重新编码的情况。
