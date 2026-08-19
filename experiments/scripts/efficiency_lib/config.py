"""Fixed benchmark configuration for LAP efficiency experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingAnchorConfig:
    model: str
    task: str
    dataset_file: Path
    checkpoint: Path
    latent_cache: Path
    training_latent_cache: Path
    partition_dir: Path
    gate_manifest: Path
    data_file: Path
    batch_size: int
    num_workers: int
    cpu_threads: int
    precision: str
    history_size: int
    num_preds: int
    frameskip: int
    img_size: int
    train_fraction: float
    split_seed: int
    seed: int
    lap_epochs_per_expert: int
    joint_epochs: int


@dataclass(frozen=True)
class InferenceTaskConfig:
    task: str
    config_name: str
    dataset_tag: str
    checkpoint: Path
    lap_run_dir: Path
    gate_manifest: Path
    eval_starts: Path | None


REPO_ROOT = Path("/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models")


ANCHOR_TRAINING = TrainingAnchorConfig(
    model="lewm",
    task="tworoom",
    dataset_file=Path("/data/sicong/weitao/datasets/lewm/tworoom.h5"),
    checkpoint=Path("/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt"),
    latent_cache=REPO_ROOT
    / "experiments/tworoom/results/auto_gate_complete_k3/tworoom_lewm_train_latent_cache.npz",
    training_latent_cache=REPO_ROOT
    / "experiments/tworoom/subjepa/formal/preparation/embedding_cache.npz",
    partition_dir=REPO_ROOT
    / "experiments/tworoom/results/auto_gate_complete_k3/auto/partition",
    gate_manifest=REPO_ROOT
    / "experiments/tworoom/results/auto_gate_complete_k3/auto/partition/manifest.json",
    data_file=Path("/data/sicong/weitao/datasets/lewm/tworoom.h5"),
    batch_size=128,
    num_workers=4,
    cpu_threads=4,
    precision="fp32",
    history_size=3,
    num_preds=1,
    frameskip=5,
    img_size=224,
    train_fraction=0.9,
    split_seed=3072,
    seed=42,
    lap_epochs_per_expert=1,
    joint_epochs=3,
)


def inference_tasks(repo_root: Path) -> dict[str, InferenceTaskConfig]:
    ckpt_root = Path("/data/sicong/weitao/.stable_worldmodel")
    return {
        "tworoom": InferenceTaskConfig(
            task="tworoom",
            config_name="tworoom",
            dataset_tag="tworoom",
            checkpoint=ckpt_root / "tworoom/lewm_object.ckpt",
            lap_run_dir=repo_root
            / "experiments/tworoom/subjepa/training/spectral/partition0_train0",
            gate_manifest=repo_root
            / "experiments/tworoom/results/auto_gate_complete_k3/auto/partition/manifest.json",
            eval_starts=repo_root
            / "experiments/tworoom/results/tworoom_success_rate_baseline_seed42/results.json",
        ),
        "pusht": InferenceTaskConfig(
            task="pusht",
            config_name="pusht",
            dataset_tag="pusht",
            checkpoint=ckpt_root / "pusht/lewm_object.ckpt",
            lap_run_dir=repo_root / "experiments/pusht/matrix/training/global/train0",
            gate_manifest=repo_root
            / "experiments/pusht/results/auto_gate_complete_k3/auto/partition/manifest.json",
            eval_starts=repo_root
            / "experiments/pusht/matrix/eval/official/eval0/results.json",
        ),
        "reacher": InferenceTaskConfig(
            task="reacher",
            config_name="reacher",
            dataset_tag="reacher",
            checkpoint=ckpt_root / "reacher/lewm_object.ckpt",
            lap_run_dir=repo_root / "experiments/reacher/matrix/training/global/train0",
            gate_manifest=repo_root
            / "experiments/reacher/results/auto_gate_complete_k3/auto/partition/manifest.json",
            eval_starts=repo_root
            / "experiments/reacher/matrix/eval/official/eval0/results.json",
        ),
        "cube": InferenceTaskConfig(
            task="cube",
            config_name="cube",
            dataset_tag="cube",
            checkpoint=ckpt_root / "cube/lewm_object.ckpt",
            lap_run_dir=repo_root / "experiments/cube/matrix/training/global/train0",
            gate_manifest=repo_root
            / "experiments/cube/results/auto_gate_complete_k3/auto/partition/manifest.json",
            eval_starts=repo_root
            / "experiments/cube/matrix/eval/official/eval0/results.json",
        ),
    }


INFERENCE_TASKS = inference_tasks(REPO_ROOT)
