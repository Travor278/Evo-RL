#!/usr/bin/env python

from lerobot.scripts.lerobot_value_train import should_save_value_checkpoint


def test_value_checkpoint_schedule_supports_explicit_steps_and_final_step():
    selected = [
        step
        for step in range(1, 8001)
        if should_save_value_checkpoint(step, total_steps=8000, save_freq=8000, save_steps=[1000, 5000])
    ]
    assert selected == [1000, 5000, 8000]
