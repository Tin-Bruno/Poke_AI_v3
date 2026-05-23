from __future__ import annotations

import unittest

from scripts.eval_freeplay import REWARD_TRACE_FIELDS, TRACE_FIELDNAMES, trace_row


class EvalFreeplayTraceTest(unittest.TestCase):
    def test_trace_row_includes_reward_breakdown(self) -> None:
        row = trace_row(
            {
                "step_count": 1,
                "map_id": 12,
                "x": 10,
                "y": 35,
                "action": "right",
                "reward_coord": 0.02,
                "reward_repeat_penalty": -0.002,
                "reward_step_penalty": -0.001,
                "steps_since_progress": 3,
                "freeplay_progress": 1.0,
            },
            reward=0.017,
        )

        for field in TRACE_FIELDNAMES:
            self.assertIn(field, row)
        for field in REWARD_TRACE_FIELDS:
            self.assertIn(field, row)
        self.assertEqual(row["reward_coord"], 0.02)
        self.assertEqual(row["reward_repeat_penalty"], -0.002)
        self.assertEqual(row["reward_step_penalty"], -0.001)
        self.assertEqual(row["steps_since_progress"], 3)
        self.assertEqual(row["freeplay_progress"], 1.0)


if __name__ == "__main__":
    unittest.main()
