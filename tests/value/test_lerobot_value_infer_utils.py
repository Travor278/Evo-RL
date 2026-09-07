#!/usr/bin/env python

import numpy as np

from lerobot.scripts.lerobot_value_infer import (
    _binarize_advantages,
    _compute_dense_rewards_from_targets,
    _compute_n_step_advantages,
    _compute_task_thresholds,
)


def test_compute_dense_rewards_from_targets_terminal_handling():
    episode_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    frame_indices = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    targets = np.array([-0.6, -0.4, -0.2, -0.8, -0.5], dtype=np.float32)

    rewards = _compute_dense_rewards_from_targets(targets, episode_indices, frame_indices)
    expected = np.array([-0.2, -0.2, -0.2, -0.3, -0.5], dtype=np.float32)
    assert np.allclose(rewards, expected)


def test_compute_n_step_advantages_simple_case():
    rewards = np.array([-0.2, -0.2, -0.2], dtype=np.float32)
    values = np.array([-0.5, -0.3, -0.1], dtype=np.float32)
    episode_indices = np.array([0, 0, 0], dtype=np.int64)
    frame_indices = np.array([0, 1, 2], dtype=np.int64)

    advantages = _compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=2,
    )

    expected = np.array([0.0, -0.1, -0.1], dtype=np.float32)
    assert np.allclose(advantages, expected)


def test_compute_task_thresholds_and_binarize_with_interventions():
    task_indices = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    advantages = np.array([-0.4, -0.1, 0.3, -0.2, 0.2], dtype=np.float32)
    interventions = np.array([0, 1, 0, 0, 0], dtype=np.float32)

    thresholds = _compute_task_thresholds(task_indices, advantages, positive_ratio=0.5)

    indicators = _binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=True,
    )

    assert indicators.tolist() == [0, 1, 1, 0, 1]


def test_attempt_transition_mask_stops_n_step_and_disables_bootstrap():
    rewards = np.array([-0.2, -0.2, -0.2, -0.2], dtype=np.float32)
    values = np.array([-0.5, -0.3, -0.1, 0.0], dtype=np.float32)
    episode_indices = np.zeros(4, dtype=np.int64)
    frame_indices = np.arange(4, dtype=np.int64)
    transition_valid = np.array([1, 1, 0, 1], dtype=np.bool_)

    advantages = _compute_n_step_advantages(
        rewards=rewards,
        values=values,
        episode_indices=episode_indices,
        frame_indices=frame_indices,
        n_step=3,
        transition_valid=transition_valid,
    )

    assert np.allclose(advantages[:3], np.array([0.1, 0.1, 0.1], dtype=np.float32))


def test_threshold_mask_uses_train_only_and_apply_mask_leaves_untagged_samples():
    task_indices = np.zeros(5, dtype=np.int64)
    advantages = np.array([-1.0, 0.0, 100.0, 200.0, 300.0], dtype=np.float32)
    train_mask = np.array([1, 1, 0, 0, 0], dtype=np.bool_)
    apply_mask = np.array([1, 1, 1, 0, 0], dtype=np.bool_)
    interventions = np.array([0, 0, 1, 1, 1], dtype=np.float32)

    thresholds = _compute_task_thresholds(
        task_indices,
        advantages,
        positive_ratio=0.5,
        threshold_mask=train_mask,
    )
    assert thresholds == {0: -0.5}
    indicators = _binarize_advantages(
        task_indices=task_indices,
        advantages=advantages,
        thresholds=thresholds,
        interventions=interventions,
        force_intervention_positive=True,
        apply_mask=apply_mask,
    )
    assert indicators.tolist() == [0, 1, 1, 0, 0]
