from __future__ import annotations

import unittest

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig, get_phase
from rewards import make_reward
from rewards.dialog_reward import DialogReward
from rewards.freeplay_reward import FREEPLAY_RAM_SHAPE, FreeplayReward, snapshot_vector
from rewards.party_reward import PartyReward
from rewards.target_position_reward import TargetPositionReward
from rewards.waypoint_reward import WaypointReward


def snap(
    map_id: int = 1,
    x: int = 1,
    y: int = 1,
    event_count: int = 0,
    hp_fraction: float = 1.0,
    party_levels: tuple[int, ...] = (),
    party_species: tuple[int, ...] = (),
    in_battle: bool = False,
) -> GameSnapshot:
    return GameSnapshot(
        map_id=map_id,
        x=x,
        y=y,
        badges=0,
        party_count=len(party_levels),
        party_levels=party_levels,
        party_species=party_species,
        hp_fraction=hp_fraction,
        in_battle=in_battle,
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

    def test_target_position_success_reward_only_once(self) -> None:
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
        reward = TargetPositionReward(closer_reward=0.1, success_reward=5.0)
        reward.reset(snap(x=4, y=5), phase)

        first = reward.step(snap(x=5, y=5), phase)
        repeated = reward.step(snap(x=5, y=5), phase)

        self.assertGreaterEqual(first.value, 5.0)
        self.assertTrue(first.progress)
        self.assertEqual(repeated.value, 0.0)
        self.assertFalse(repeated.progress)

    def test_waypoint_reward_advances_to_next_point(self) -> None:
        phase = PhaseConfig(
            id="test",
            name="Teste waypoints",
            state="test.state",
            model="models/test.zip",
            max_steps=10,
            rewards=("waypoint",),
            success="event_count_increase",
            waypoints=((1, 3, 1), (1, 4, 1)),
        )
        reward = WaypointReward(closer_reward=0.1, reached_reward=1.0)
        reward.reset(snap(map_id=1, x=1, y=1), phase)

        closer = reward.step(snap(map_id=1, x=2, y=1), phase)
        reached = reward.step(snap(map_id=1, x=3, y=1), phase)

        self.assertGreater(closer.value, 0)
        self.assertEqual(reached.terms["waypoint_index"], 1)
        self.assertGreaterEqual(reached.value, 1.0)

    def test_party_reward_when_party_count_increases(self) -> None:
        phase = get_phase("phase5c")
        reward = PartyReward(gained_reward=10.0)
        reward.reset(snap(party_levels=()), phase)

        result = reward.step(snap(party_levels=(5,)), phase)

        self.assertEqual(result.value, 10.0)
        self.assertTrue(result.progress)

    def test_freeplay_reward_tracks_general_progress(self) -> None:
        reward = FreeplayReward()
        reward.reset(snap(map_id=0, x=1, y=1, event_count=2, party_levels=(5,), party_species=(1,)))

        value, progress, terms = reward.score(
            snap(map_id=12, x=4, y=4, event_count=4, party_levels=(6,), party_species=(1,))
        )

        self.assertGreater(value, 0.0)
        self.assertTrue(progress)
        self.assertGreater(terms["reward_map"], 0.0)
        self.assertGreater(terms["reward_events"], 0.0)
        self.assertGreater(terms["reward_party"], 0.0)

    def test_freeplay_reward_penalizes_repeated_exact_coord(self) -> None:
        reward = FreeplayReward(same_coord_limit=2, same_coord_penalty=0.5, step_penalty=0.0)
        current = snap(map_id=12, x=10, y=35)
        reward.reset(current)

        value, progress, terms = reward.score(current)

        self.assertLess(value, 0.0)
        self.assertFalse(progress)
        self.assertLess(terms["reward_stuck_penalty"], 0.0)
        self.assertEqual(terms["coord_visit_count"], 2.0)

    def test_freeplay_reward_tracks_battle_result_and_faint(self) -> None:
        reward = FreeplayReward(step_penalty=0.0, battle_win_reward=2.0, faint_penalty=3.0)
        reward.reset(snap(map_id=12, x=1, y=1, hp_fraction=0.5, party_levels=(5,), in_battle=True))

        win_value, win_progress, win_terms = reward.score(
            snap(map_id=12, x=1, y=1, hp_fraction=0.4, party_levels=(5,), in_battle=False)
        )

        reward.reset(snap(map_id=12, x=1, y=1, hp_fraction=0.5, party_levels=(5,), in_battle=True))
        faint_value, faint_progress, faint_terms = reward.score(
            snap(map_id=12, x=1, y=1, hp_fraction=0.0, party_levels=(5,), in_battle=True)
        )

        self.assertGreater(win_value, 0.0)
        self.assertTrue(win_progress)
        self.assertGreater(win_terms["reward_battle"], 0.0)
        self.assertLess(faint_value, 0.0)
        self.assertFalse(faint_progress)
        self.assertLess(faint_terms["reward_faint_penalty"], 0.0)

    def test_freeplay_snapshot_vector_has_stable_shape(self) -> None:
        vector = snapshot_vector(snap(party_levels=(5,), party_species=(1,)))

        self.assertEqual(vector.shape, FREEPLAY_RAM_SHAPE)
        self.assertEqual(vector.dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()
