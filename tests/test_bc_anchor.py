"""Tests for the BC-anchor term in hier_ppo.ppo_update: an auxiliary
demonstration-NLL loss that stops RL fine-tuning from destroying cloned
skills (observed directly: naive PPO fine-tuning dragged a BC policy from
12% to 5% win rate vs heuristic)."""
import numpy as np
import torch

from env.pettingzoo_env import CatanAECEnv
from scripts.collect_heuristic_demos import play_and_record
from training.hier_model import HierarchicalActorCritic
from training.hier_ppo import collect_rollout, compute_holdout_nll, ppo_update
from training.model import observation_dim


def _demo_tensors(n_games: int = 2) -> dict[str, torch.Tensor]:
    records = []
    for seed in range(n_games):
        records.extend(play_and_record(seed, public_hand_features=False))
    return {
        "obs": torch.as_tensor(np.stack([r["obs"] for r in records]), dtype=torch.float32),
        "type_mask": torch.as_tensor(np.stack([r["type_mask"] for r in records])),
        "type_idx": torch.as_tensor(np.array([r["type_idx"] for r in records], dtype=np.int64)),
        "head1_id": torch.as_tensor(np.array([r["head1_id"] for r in records], dtype=np.int64)),
        "sub_mask_1": torch.as_tensor(np.stack([r["sub_mask_1"] for r in records])),
        "sub_idx_1": torch.as_tensor(np.array([r["sub_idx_1"] for r in records], dtype=np.int64)),
        "head2_id": torch.as_tensor(np.array([r["head2_id"] for r in records], dtype=np.int64)),
        "sub_mask_2": torch.as_tensor(np.stack([r["sub_mask_2"] for r in records])),
        "sub_idx_2": torch.as_tensor(np.array([r["sub_idx_2"] for r in records], dtype=np.int64)),
    }


def _demo_logprob(model, demos) -> float:
    with torch.inference_mode():
        logprob, _, _ = model.evaluate_actions(
            demos["obs"], {k: v for k, v in demos.items() if k != "obs"})
    return float(logprob.mean())


def test_bc_anchor_pulls_policy_toward_demonstrations():
    torch.manual_seed(0)
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)
    demos = _demo_tensors()
    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=80)
    transitions, _ = collect_rollout(env, model, "cpu", num_episodes=2, base_seed=0)

    before = _demo_logprob(model, demos)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = ppo_update(model, optimizer, transitions, epochs=2, minibatch_size=64,
                       bc_dataset=demos, bc_coef=5.0, bc_minibatch_size=256, target_kl=None)
    after = _demo_logprob(model, demos)

    assert "bc_loss" in stats and np.isfinite(stats["bc_loss"])
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} not finite: {v}"
    assert after > before, (
        f"strong anchor should raise demo logprob (before={before:.4f}, after={after:.4f})")


def test_annealed_bc_coef_schedule():
    from training.league_train import annealed_bc_coef

    # no annealing: constant
    assert annealed_bc_coef(0.2, None, 500, 1, 600) == 0.2
    # endpoints and midpoint of a fresh 600-iter run (iters 1..600)
    assert abs(annealed_bc_coef(0.2, 0.02, 1, 1, 600) - 0.2) < 1e-9
    assert abs(annealed_bc_coef(0.2, 0.02, 600, 1, 600) - 0.02) < 1e-9
    mid = annealed_bc_coef(0.2, 0.02, 300, 1, 600)
    assert 0.10 < mid < 0.12
    # resume-aware: a resumed run starting at 401 anneals over ITS OWN span
    assert abs(annealed_bc_coef(0.2, 0.02, 401, 401, 200) - 0.2) < 1e-9
    assert abs(annealed_bc_coef(0.2, 0.02, 600, 401, 200) - 0.02) < 1e-9
    # never over/undershoots
    assert abs(annealed_bc_coef(0.2, 0.02, 9999, 1, 600) - 0.02) < 1e-9


def test_compute_holdout_nll_matches_direct_evaluate_actions():
    torch.manual_seed(2)
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)
    demos = _demo_tensors()

    chunked = compute_holdout_nll(model, demos, batch_size=97)  # odd size forces multiple uneven chunks
    with torch.inference_mode():
        logprob, _, _ = model.evaluate_actions(demos["obs"], {k: v for k, v in demos.items() if k != "obs"})
    direct = -float(logprob.mean())

    assert abs(chunked - direct) < 1e-4
    assert np.isfinite(chunked)


def test_compute_holdout_nll_rises_when_policy_is_pushed_away_from_demos():
    """The whole point of the holdout signal: it should detect drift away
    from demonstrated behavior, as a proxy for the erosion observed in
    long low-anchor-coefficient runs."""
    torch.manual_seed(3)
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)
    demos = _demo_tensors()
    before = compute_holdout_nll(model, demos)

    # Push the policy hard AWAY from the demonstrated actions (the opposite
    # of BC) by maximizing their NLL for a few steps -- simulates unanchored
    # drift without needing a long rollout-based training loop.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(20):
        logprob, _, _ = model.evaluate_actions(demos["obs"], {k: v for k, v in demos.items() if k != "obs"})
        loss = logprob.mean()  # ascend NLL == descend logprob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after = compute_holdout_nll(model, demos)
    assert after > before, f"expected drift to raise holdout NLL (before={before:.3f}, after={after:.3f})"


def test_ppo_update_without_anchor_is_unchanged():
    torch.manual_seed(1)
    model = HierarchicalActorCritic(obs_dim=observation_dim(), hidden=32)
    env = CatanAECEnv(randomize_board=False, allow_trading=False, allow_dev_cards=False,
                       max_episode_steps=80)
    transitions, _ = collect_rollout(env, model, "cpu", num_episodes=2, base_seed=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = ppo_update(model, optimizer, transitions, epochs=1, minibatch_size=64)
    assert "bc_loss" not in stats
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} not finite: {v}"
