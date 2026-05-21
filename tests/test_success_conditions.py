from __future__ import annotations

import unittest

from envs.success_conditions import never, target_position_after_event_count
from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig


def snap(map_id: int = 40, x: int = 5, y: int = 3, event_count: int = 1) -> GameSnapshot:
    return GameSnapshot(
        map_id=map_id,
        x=x,
        y=y,
        badges=0,
        party_count=0,
        party_levels=(),
        event_count=event_count,
    )


class SuccessConditionsTest(unittest.TestCase):
    def test_target_position_after_event_rejects_early_path_tile(self) -> None:
        phase = PhaseConfig(
            id="test",
            name="Teste",
            state="test.state",
            model="models/test.zip",
            max_steps=10,
            rewards=(),
            success="target_position_after_event_count",
            target_map=40,
            target_x=5,
            target_y=3,
            target_event_count_delta=3,
        )

        self.assertFalse(target_position_after_event_count(snap(), snap(), phase))
        self.assertFalse(target_position_after_event_count(snap(event_count=3), snap(), phase))
        self.assertFalse(target_position_after_event_count(snap(y=4, event_count=4), snap(), phase))
        self.assertTrue(target_position_after_event_count(snap(event_count=4), snap(), phase))

    def test_never_success_condition_does_not_end_freeplay(self) -> None:
        phase = PhaseConfig(
            id="freeplay",
            name="Freeplay",
            state="freeplay.state",
            model="models/freeplay.zip",
            max_steps=10,
            rewards=(),
            success="never",
        )

        self.assertFalse(never(snap(), snap(), phase))


if __name__ == "__main__":
    unittest.main()
