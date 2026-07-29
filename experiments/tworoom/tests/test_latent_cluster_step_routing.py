from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from jepa import JEPA
from tworoom_success_rate_eval import LatentClusterSwitchJEPA


class DummyEncoder(nn.Module):
    def forward(self, pixels, interpolate_pos_encoding=True):
        del interpolate_pos_encoding
        return SimpleNamespace(last_hidden_state=pixels.unsqueeze(1))


class IdentityActionEncoder(nn.Module):
    def forward(self, action):
        return action


class ShiftPredictor(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def forward(self, emb, act_emb):
        del act_emb
        return emb + self.delta.view(1, 1, -1)


def _jepa(encoder, action_encoder, delta):
    return JEPA(
        encoder=encoder,
        predictor=ShiftPredictor(delta),
        action_encoder=action_encoder,
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )


def test_step_routing_switches_expert_from_predicted_latent():
    encoder = DummyEncoder()
    action_encoder = IdentityActionEncoder()
    base = _jepa(encoder, action_encoder, [-2.0, 0.0])
    cluster_models = {
        "cluster0": _jepa(encoder, action_encoder, [-2.0, 0.0]),
        "cluster1": _jepa(encoder, action_encoder, [2.0, 0.0]),
        "cluster2": _jepa(encoder, action_encoder, [0.0, 0.0]),
    }
    model = LatentClusterSwitchJEPA(
        base,
        cluster_models,
        centroids=torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ).numpy(),
        spherical=True,
        routing_mode="step",
    )

    info = {
        "pixels": torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]],
            dtype=torch.float32,
        )
    }
    actions = torch.zeros((1, 2, 3, 1), dtype=torch.float32)
    result = model.rollout(info, actions, history_size=1)

    expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
    torch.testing.assert_close(result["predicted_emb"][0, 0, :, 0], expected)
    torch.testing.assert_close(result["predicted_emb"][0, 1, :, 0], expected)
    assert model.route_histogram.tolist() == [4, 2, 0]
    assert model.route_call_count == 3
    assert model.route_assignment_count == 6
    assert model.route_transition_count.item() == 4
    assert model.route_switch_count.item() == 4


def test_step_history3_routes_from_latest_predicted_state():
    """history_size is context capacity; step routing follows current state.

    Online TwoRoom starts with one observed latent.  Even while the recurrent
    context grows toward three entries, the next expert must be selected from
    the latest predicted state rather than the oldest entry in that context.
    """

    encoder = DummyEncoder()
    action_encoder = IdentityActionEncoder()
    base = _jepa(encoder, action_encoder, [-2.0, 0.0])
    cluster_models = {
        "cluster0": _jepa(encoder, action_encoder, [-2.0, 0.0]),
        "cluster1": _jepa(encoder, action_encoder, [2.0, 0.0]),
    }
    model = LatentClusterSwitchJEPA(
        base,
        cluster_models,
        centroids=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]).numpy(),
        spherical=True,
        routing_mode="step",
    )
    info = {
        "pixels": torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]]]], dtype=torch.float32
        )
    }
    actions = torch.zeros((1, 1, 4, 1), dtype=torch.float32)
    result = model.rollout(info, actions, history_size=3)

    torch.testing.assert_close(
        result["predicted_emb"][0, 0, :, 0],
        torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]),
    )
    assert model.route_histogram.tolist() == [2, 2]
    assert model.route_switch_count.item() == 3


def test_zscore_l2_torch_transform_matches_definition():
    encoder = DummyEncoder()
    action_encoder = IdentityActionEncoder()
    base = _jepa(encoder, action_encoder, [0.0, 0.0])
    cluster_models = {
        f"cluster{i}": _jepa(encoder, action_encoder, [0.0, 0.0])
        for i in range(3)
    }
    model = LatentClusterSwitchJEPA(
        base,
        cluster_models,
        centroids=torch.eye(3, 2).numpy(),
        spherical=True,
        zscore={
            "mu": torch.tensor([1.0, 2.0]).numpy(),
            "sigma": torch.tensor([2.0, 4.0]).numpy(),
            "eps": 0.5,
        },
        routing_mode="step",
    )
    latent = torch.tensor([[6.0, -2.5], [-1.5, 11.0]])
    expected = torch.nn.functional.normalize(
        (latent - torch.tensor([1.0, 2.0]))
        / (torch.tensor([2.0, 4.0]) + 0.5),
        dim=1,
    )
    torch.testing.assert_close(model._transform_latent(latent), expected)


def test_mpc_routes_each_environment_and_reuses_route_across_cem_calls():
    encoder = DummyEncoder()
    action_encoder = IdentityActionEncoder()
    base = _jepa(encoder, action_encoder, [0.0, 0.0])
    cluster_models = {
        "cluster0": _jepa(encoder, action_encoder, [-2.0, 0.0]),
        "cluster1": _jepa(encoder, action_encoder, [2.0, 0.0]),
    }
    model = LatentClusterSwitchJEPA(
        base,
        cluster_models,
        centroids=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]).numpy(),
        spherical=True,
        routing_mode="mpc",
    )
    pixels = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[-1.0, 0.0]], [[-1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    actions = torch.zeros((2, 2, 2, 1), dtype=torch.float32)
    first = model.rollout({"pixels": pixels}, actions, history_size=1)
    second = model.rollout({"pixels": pixels}, actions, history_size=1)

    torch.testing.assert_close(
        first["predicted_emb"][:, :, 1, 0],
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    torch.testing.assert_close(
        second["predicted_emb"][:, :, 1, 0],
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
    )
    assert model.classify_count == 1
    assert model.classify_assignment_count == 2
    assert model.mpc_route_cache_misses == 1
    assert model.mpc_route_cache_hits == 1


def test_tensor_identity_supports_inference_tensors():
    normal = torch.zeros(2, 3)
    normal_before = LatentClusterSwitchJEPA._tensor_identity(normal)
    normal.add_(1)
    normal_after = LatentClusterSwitchJEPA._tensor_identity(normal)
    assert normal_before != normal_after

    with torch.inference_mode():
        inference = torch.zeros(2, 3)
        first = LatentClusterSwitchJEPA._tensor_identity(inference)
        second = LatentClusterSwitchJEPA._tensor_identity(inference)
    assert inference.is_inference()
    assert first == second
    assert first[-1] is None


if __name__ == "__main__":
    test_step_routing_switches_expert_from_predicted_latent()
    test_step_history3_routes_from_latest_predicted_state()
    test_zscore_l2_torch_transform_matches_definition()
    test_mpc_routes_each_environment_and_reuses_route_across_cem_calls()
    test_tensor_identity_supports_inference_tensors()
    print("5 routing tests passed")
