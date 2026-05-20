from __future__ import annotations

import unittest

from envs import PokemonRedEnv
from phases import get_phase
from phases.phase_config import (
    DIALOG_ACTIONS,
    DIALOG_CONFIRM_ACTIONS,
    DIALOG_RELEASE_ACTIONS,
    MOVE_ACTIONS,
)


class EnvConfigTest(unittest.TestCase):
    def test_coords_observation_uses_mlp_friendly_space(self) -> None:
        env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase1"),
            observation_mode="coords",
        )

        self.assertEqual(env.observation_space.shape, (3,))
        self.assertEqual(env.observation_space.dtype.name, "uint8")

    def test_ram_observation_adds_event_and_party_state(self) -> None:
        env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=get_phase("phase5"),
            observation_mode="ram",
        )

        self.assertEqual(env.observation_space.shape, (6,))
        self.assertEqual(env.observation_space.dtype.name, "float32")

    def test_movement_phases_use_small_action_set(self) -> None:
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

        self.assertEqual(env.actions, MOVE_ACTIONS)
        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(phase2_env.actions, MOVE_ACTIONS)
        self.assertEqual(phase2_env.action_space.n, 5)
        self.assertEqual(phase3_env.actions, MOVE_ACTIONS)
        self.assertEqual(phase3_env.action_space.n, 5)

    def test_dialog_phases_use_b_instead_of_a_when_no_confirm_is_needed(self) -> None:
        self.assertEqual(get_phase("phase6").actions, DIALOG_ACTIONS)

    def test_phase4_uses_a_and_b_for_oak_event_confirmation(self) -> None:
        self.assertEqual(get_phase("phase4").actions, DIALOG_CONFIRM_ACTIONS)

    def test_phase5_uses_b_then_down_to_prove_control(self) -> None:
        self.assertEqual(get_phase("phase5").actions, DIALOG_RELEASE_ACTIONS)

    def test_phase5b_uses_only_movement_to_reach_pokeball(self) -> None:
        self.assertEqual(get_phase("phase5b").actions, MOVE_ACTIONS)

    def test_phase5c_uses_a_and_b_to_choose_starter(self) -> None:
        self.assertEqual(get_phase("phase5c").actions, DIALOG_CONFIRM_ACTIONS)

    def test_phase3_success_requires_event_progress(self) -> None:
        phase = get_phase("phase3")

        self.assertEqual(phase.success, "event_count_increase")
        self.assertEqual(phase.target_map, 0)
        self.assertEqual((phase.target_x, phase.target_y), (10, 1))
        self.assertEqual(phase.rewards, ("waypoint", "dialog"))
        self.assertEqual(phase.waypoints, ((0, 8, 2), (0, 10, 2), (0, 10, 1)))

    def test_phase4_success_requires_entering_lab(self) -> None:
        phase = get_phase("phase4")
        env = PokemonRedEnv(
            rom_path="missing.gb",
            phase=phase,
            observation_mode="coords",
        )

        self.assertEqual(phase.success, "target_map")
        self.assertEqual(phase.rewards, ("target_map", "dialog"))
        self.assertEqual(phase.target_map, 40)
        self.assertIsNone(phase.target_x)
        self.assertIsNone(phase.target_y)
        self.assertEqual(env.actions, DIALOG_CONFIRM_ACTIONS)

    def test_phase5_stops_after_initial_lab_dialog(self) -> None:
        phase = get_phase("phase5")

        self.assertEqual(phase.success, "target_position_after_event_count")
        self.assertEqual(phase.rewards, ("dialog", "target_position"))
        self.assertEqual((phase.target_x, phase.target_y), (5, 3))
        self.assertEqual(phase.target_event_count_delta, 3)
        self.assertEqual(phase.waypoints, ())
        self.assertEqual(phase.actions, DIALOG_RELEASE_ACTIONS)
        self.assertEqual(phase.save_actions, ("b",))

    def test_phase5b_reaches_front_of_pokeball(self) -> None:
        phase = get_phase("phase5b")

        self.assertEqual(phase.success, "target_position")
        self.assertEqual(phase.rewards, ("waypoint", "target_position"))
        self.assertEqual((phase.target_x, phase.target_y), (6, 4))
        self.assertEqual(phase.waypoints, ())
        self.assertEqual(phase.actions, MOVE_ACTIONS)
        self.assertEqual(phase.save_actions, ("up",))

    def test_phase5c_rewards_starter_pickup(self) -> None:
        phase = get_phase("phase5c")

        self.assertEqual(phase.success, "party_count_increase")
        self.assertEqual(phase.rewards, ("dialog", "party"))
        self.assertEqual(phase.actions, DIALOG_CONFIRM_ACTIONS)

    def test_phase8_stops_after_post_battle_dialog_and_control(self) -> None:
        phase = get_phase("phase8")

        self.assertEqual(phase.success, "target_position")
        self.assertEqual(phase.rewards, ("target_position",))
        self.assertEqual((phase.target_x, phase.target_y), (5, 7))
        self.assertEqual(phase.actions, DIALOG_RELEASE_ACTIONS)
        self.assertEqual(phase.scripted_actions, ("b",) * 20 + ("down",) * 4)


if __name__ == "__main__":
    unittest.main()
