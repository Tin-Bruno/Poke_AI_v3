from __future__ import annotations

import unittest

from poke_ai_v3.memory import GameSnapshot
from poke_ai_v3.rewards import ProgressReward, RewardConfig


def snap(
    map_id: int = 1,
    x: int = 1,
    y: int = 1,
    badges: int = 0,
    party_count: int = 0,
    party_levels: tuple[int, ...] = (),
    hp_fraction: float = 0.0,
    event_count: int = 0,
    max_opponent_level: int = 0,
) -> GameSnapshot:
    return GameSnapshot(
        map_id=map_id,
        x=x,
        y=y,
        badges=badges,
        party_count=party_count,
        party_levels=party_levels,
        hp_fraction=hp_fraction,
        event_count=event_count,
        max_opponent_level=max_opponent_level,
    )


class ProgressRewardTest(unittest.TestCase):
    def test_rewards_new_position(self) -> None:
        rewarder = ProgressReward(RewardConfig(step_penalty=0.0, new_position_reward=1.0))
        rewarder.reset(snap(x=1))

        reward, terminated, truncated, info = rewarder.step(snap(x=2))

        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["visited_positions"], 2)

    def test_rewards_badge_gain_once(self) -> None:
        rewarder = ProgressReward(RewardConfig(step_penalty=0.0, badge_reward=10.0))
        rewarder.reset(snap(badges=0))

        reward, *_ = rewarder.step(snap(badges=0b00000011))
        reward_again, *_ = rewarder.step(snap(badges=0b00000011))

        self.assertEqual(reward, 20.0)
        self.assertEqual(reward_again, 0.0)

    def test_truncates_when_stuck(self) -> None:
        rewarder = ProgressReward(RewardConfig(step_penalty=0.0, max_no_progress_steps=2))
        current = snap()
        rewarder.reset(current)

        rewarder.step(current)
        _, _, truncated, info = rewarder.step(current)

        self.assertTrue(truncated)
        self.assertEqual(info["steps_since_progress"], 2)

    def test_rewards_event_gain(self) -> None:
        rewarder = ProgressReward(RewardConfig(step_penalty=0.0, event_reward=4.0))
        rewarder.reset(snap(event_count=10))

        reward, *_ = rewarder.step(snap(event_count=13))

        self.assertEqual(reward, 12.0)

    def test_rewards_healing_delta_squared(self) -> None:
        rewarder = ProgressReward(RewardConfig(step_penalty=0.0, heal_reward=10.0))
        rewarder.reset(snap(hp_fraction=0.5, party_count=1))

        reward, *_ = rewarder.step(snap(hp_fraction=0.7, party_count=1))

        self.assertAlmostEqual(reward, 0.4)


if __name__ == "__main__":
    unittest.main()
