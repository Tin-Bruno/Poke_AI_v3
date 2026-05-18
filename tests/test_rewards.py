from __future__ import annotations

import unittest

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig, get_phase
from rewards import make_reward
from rewards.dialog_reward import DialogReward
from rewards.target_position_reward import TargetPositionReward


def snap(
    map_id: int = 1,
    x: int = 1,
    y: int = 1,
    event_count: int = 0,
    hp_fraction: float = 1.0,
    party_levels: tuple[int, ...] = (),
) -> GameSnapshot:
    return GameSnapshot(
        map_id=map_id,
        x=x,
        y=y,
        badges=0,
        party_count=len(party_levels),
        party_levels=party_levels,
        hp_fraction=hp_fraction,
        event_count=event_count,
    )


class PhaseRewardTest(unittest.TestCase):
    def test_phase1_rewards_target_map(self) -> None:
        rewarder = make_reward(get_phase("phase1"), step_penalty=0.0, scale=1.0)
        rewarder.reset(snap(map_id=1))

        reward, truncated, terms = rewarder.step(snap(map_id=37))

        self.assertGreaterEqual(reward, 10.0)
        self.assertFalse(truncated)
        self.assertGreater(terms["target_map"], 0)

    def test_reward_timeout_when_no_progress(self) -> None:
        phase = PhaseConfig(
            id="test",
            name="Teste",
            state="test.state",
            model="models/test.zip",
            max_steps=10,
            rewards=("new_map",),
            success="target_map",
            target_map=2,
        )
        current = snap(map_id=1)
        rewarder = make_reward(phase, step_penalty=0.0, max_no_progress_steps=2)
        rewarder.reset(current)

        rewarder.step(current)
        _, truncated, terms = rewarder.step(current)

        self.assertTrue(truncated)
        self.assertEqual(terms["steps_since_progress"], 2)

    def test_dialog_reward_event_gain(self) -> None:
        phase = get_phase("phase3")
        reward = DialogReward(event_reward=2.0)
        reward.reset(snap(event_count=10), phase)

        result = reward.step(snap(event_count=13), phase)

        self.assertEqual(result.value, 6.0)
        self.assertTrue(result.progress)

    def test_target_position_gets_closer(self) -> None:
        phase = PhaseConfig(
            id="test",
            name="Teste posicao",
            state="test.state",
            model="models/test.zip",
            max_steps=10,
            rewards=("target_position",),
            success="target_position",
            target_map=1,
            target_x=5,
            target_y=5,
        )
        reward = TargetPositionReward(closer_reward=0.1)
        reward.reset(snap(x=1, y=1), phase)

        result = reward.step(snap(x=3, y=3), phase)

        self.assertGreater(result.value, 0)
        self.assertTrue(result.progress)


if __name__ == "__main__":
    unittest.main()
