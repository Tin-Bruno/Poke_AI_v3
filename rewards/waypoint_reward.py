"""Reward sequencial para guiar rotas com curvas obrigatorias."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class WaypointReward(RewardComponent):
    def __init__(self, closer_reward: float = 0.12, reached_reward: float = 1.0) -> None:
        self.closer_reward = closer_reward
        self.reached_reward = reached_reward
        self.index = 0
        self.best_distance: int | None = None

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.index = 0
        self.best_distance = self._distance_to_current(snapshot, phase)

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        if not phase.waypoints or self.index >= len(phase.waypoints):
            return RewardResult(0.0, terms={"waypoint": 0.0, "waypoint_index": self.index})

        distance = self._distance_to_current(snapshot, phase)
        reward = 0.0
        progress = False

        if distance is not None and (self.best_distance is None or distance < self.best_distance):
            reward += self.closer_reward * (self.best_distance - distance if self.best_distance else 1)
            self.best_distance = distance
            progress = True

        if distance == 0:
            reward += self.reached_reward
            progress = True
            self.index += 1
            self.best_distance = self._distance_to_current(snapshot, phase)

        return RewardResult(reward, progress, {"waypoint": reward, "waypoint_index": self.index})

    def _distance_to_current(self, snapshot: GameSnapshot, phase: PhaseConfig) -> int | None:
        if not phase.waypoints or self.index >= len(phase.waypoints):
            return None

        target_map, target_x, target_y = phase.waypoints[self.index]
        if snapshot.map_id != target_map:
            return 999
        return abs(snapshot.x - target_x) + abs(snapshot.y - target_y)
