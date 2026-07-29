from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
MODULE_DIR = THIS_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

import trajectory_switch_rollout as rollout_module  # noqa: E402
from trajectory_switch_rollout import (  # noqa: E402
    LatentPrototypeRouter,
    precompute_rooms3_keys,
    rollout_mse_latent_switch,
)


def test_prototype_index_is_mapped_through_owner_for_arbitrary_k_p() -> None:
    artifact = {
        "routing_vectors": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        # P=5 routing vectors for K=3 experts; owners are intentionally not
        # equal to prototype indices.
        "prototype_cluster_ids": np.asarray([2, 0, 1, 2, 0], dtype=np.int64),
        "meta": {"num_clusters": 3, "spherical": True},
        "zscore": None,
    }
    router = LatentPrototypeRouter(artifact, torch.device("cpu"))
    assigned = router.assign(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    )
    assert router.num_clusters == 3
    assert len(router.routing_vectors) == 5
    assert assigned.tolist() == [2, 0, 1]


class RecordingRouter:
    num_clusters = 2

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def assign(self, latent: torch.Tensor) -> torch.Tensor:
        self.calls.append(latent.detach().cpu().clone())
        return torch.where(
            latent[:, 0] >= 0,
            torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device),
            torch.ones(latent.shape[0], dtype=torch.long, device=latent.device),
        )


def _fake_predict_next(model, ctx_emb: torch.Tensor, ctx_action: torch.Tensor) -> torch.Tensor:
    del ctx_action
    return ctx_emb[:, -1] + model.delta.to(ctx_emb.device)


def _run_mode(mode: str) -> RecordingRouter:
    router = RecordingRouter()
    models = {
        "cluster0": SimpleNamespace(delta=torch.tensor([-2.0, 0.0])),
        "cluster1": SimpleNamespace(delta=torch.tensor([2.0, 0.0])),
    }
    # t=0 is the observed rollout start.  t=1 is deliberately far from the
    # first prediction (-1,0), making future-GT leakage detectable.
    latents = torch.tensor([[[1.0, 0.0], [5.0, 0.0], [7.0, 0.0]]])
    actions = torch.zeros((1, 2, 1))
    with patch.object(rollout_module, "predict_next", side_effect=_fake_predict_next):
        mse = rollout_mse_latent_switch(
            models,
            router,
            mode,
            latents,
            actions,
            history=1,
            max_steps=2,
            batch_size=1,
        )
    assert mse.shape == (2, 1)
    return router


def test_mpc_routes_once_from_observed_start() -> None:
    router = _run_mode("mpc")
    assert len(router.calls) == 1
    torch.testing.assert_close(router.calls[0], torch.tensor([[1.0, 0.0]]))


def test_step_reroutes_from_prediction_not_future_gt() -> None:
    router = _run_mode("step")
    assert len(router.calls) == 2
    torch.testing.assert_close(router.calls[0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(router.calls[1], torch.tensor([[-1.0, 0.0]]))
    assert not torch.equal(router.calls[1], torch.tensor([[5.0, 0.0]]))


def test_future_gt_routing_is_explicitly_oracle_only() -> None:
    router = _run_mode("oracle_gt_step")
    assert len(router.calls) == 2
    torch.testing.assert_close(router.calls[0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(router.calls[1], torch.tensor([[5.0, 0.0]]))


def _run_history3_mode(mode: str) -> RecordingRouter:
    router = RecordingRouter()
    models = {
        "cluster0": SimpleNamespace(delta=torch.tensor([-2.0, 0.0])),
        "cluster1": SimpleNamespace(delta=torch.tensor([2.0, 0.0])),
    }
    latents = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 0.0],
          [101.0, 0.0], [102.0, 0.0], [103.0, 0.0]]]
    )
    actions = torch.zeros((1, 6, 1))
    with patch.object(rollout_module, "predict_next", side_effect=_fake_predict_next):
        rollout_mse_latent_switch(
            models,
            router,
            mode,
            latents,
            actions,
            history=3,
            max_steps=4,
            batch_size=1,
        )
    return router


def test_history3_mpc_uses_transition_start_not_history_end() -> None:
    router = _run_history3_mode("mpc")
    assert len(router.calls) == 1
    torch.testing.assert_close(router.calls[0], torch.tensor([[1.0, 0.0]]))


def test_history3_step_uses_start_then_only_predictions() -> None:
    router = _run_history3_mode("step")
    assert len(router.calls) == 4
    torch.testing.assert_close(router.calls[0], torch.tensor([[1.0, 0.0]]))
    # First appended prediction is 3-2=1, whereas future GT z3 is 100.
    torch.testing.assert_close(router.calls[1], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(router.calls[2], torch.tensor([[-1.0, 0.0]]))
    torch.testing.assert_close(router.calls[3], torch.tensor([[1.0, 0.0]]))


def test_rooms3_history3_mpc_uses_index_zero() -> None:
    proprio = np.asarray([[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]])
    with patch.object(
        rollout_module,
        "geometry_rooms3_key",
        side_effect=lambda x, y, thresholds: f"x{int(x)}",
    ):
        mpc = precompute_rooms3_keys(proprio, history=3, max_steps=2, routing_mode="mpc")
        oracle = precompute_rooms3_keys(
            proprio, history=3, max_steps=2, routing_mode="oracle_gt_step"
        )
    assert mpc.tolist() == [["x10", "x10"]]
    assert oracle.tolist() == [["x10", "x40"]]


if __name__ == "__main__":
    test_prototype_index_is_mapped_through_owner_for_arbitrary_k_p()
    test_mpc_routes_once_from_observed_start()
    test_step_reroutes_from_prediction_not_future_gt()
    test_future_gt_routing_is_explicitly_oracle_only()
    test_history3_mpc_uses_transition_start_not_history_end()
    test_history3_step_uses_start_then_only_predictions()
    test_rooms3_history3_mpc_uses_index_zero()
    print("7 trajectory-switch routing tests passed")
