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

    def test_phase1_uses_small_movement_action_set(self) -> None:
        env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase1"),
            observation_mode="coords",
        )

        self.assertEqual(env.actions, ("up", "down", "left", "right", "noop"))
        self.assertEqual(env.action_space.n, 5)


if __name__ == "__main__":
    unittest.main()
