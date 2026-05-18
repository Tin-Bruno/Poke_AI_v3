from __future__ import annotations

import unittest

from envs import PokemonRedEnv
from phases import get_phase


class EnvConfigTest(unittest.TestCase):
    def test_coords_observation_uses_mlp_friendly_space(self) -> None:
        env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase1"),
            observation_mode="coords",
        )

        self.assertEqual(env.observation_space.shape, (3,))
        self.assertEqual(env.observation_space.dtype.name, "uint8")

    def test_movement_phases_use_small_action_set(self) -> None:
        expected_actions = ("up", "down", "left", "right", "noop")

        env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase1"),
            observation_mode="coords",
        )
        phase2_env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase2"),
            observation_mode="coords",
        )
        phase3_env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase3"),
            observation_mode="coords",
        )

        self.assertEqual(env.actions, expected_actions)
        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(phase2_env.actions, expected_actions)
        self.assertEqual(phase2_env.action_space.n, 5)
        self.assertEqual(phase3_env.actions, expected_actions)
        self.assertEqual(phase3_env.action_space.n, 5)

    def test_phase3_success_requires_event_progress(self) -> None:
        phase = get_phase("phase3")

        self.assertEqual(phase.success, "event_count_increase")
        self.assertEqual(phase.target_map, 0)
        self.assertEqual((phase.target_x, phase.target_y), (10, 1))
        self.assertEqual(phase.rewards, ("waypoint", "dialog"))
        self.assertEqual(phase.waypoints, ((0, 8, 2), (0, 10, 2), (0, 10, 1)))


if __name__ == "__main__":
    unittest.main()
