"""Reward densa para aproximar o agente de uma coordenada alvo."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class TargetPositionReward(RewardComponent):
    def __init__(self, closer_reward: float = 0.1, success_reward: float = 5.0) -> None:
        self.closer_reward = closer_reward
        self.success_reward = success_reward
        self.best_distance: int | None = None
        self.success_rewarded = False

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.best_distance = self._distance(snapshot, phase)
        self.success_rewarded = False

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        distance = self._distance(snapshot, phase)
        if distance is None:
            return RewardResult(0.0, terms={"target_position": 0.0})

        reward = 0.0
        progress = False
        if self.best_distance is None or distance < self.best_distance:
            reward += self.closer_reward * (self.best_distance - distance if self.best_distance else 1)
            self.best_distance = distance
            progress = True

        if distance <= phase.target_radius and not self.success_rewarded:
            reward += self.success_reward
            self.success_rewarded = True
            progress = True

        return RewardResult(reward, progress, {"target_position": reward})

    @staticmethod
    def _distance(snapshot: GameSnapshot, phase: PhaseConfig) -> int | None:
        if phase.target_x is None or phase.target_y is None:
            return None
        target_map = phase.target_position_map
        if target_map is None:
            target_map = phase.target_map
        if target_map is not None and snapshot.map_id != target_map:
            return 999
        return abs(snapshot.x - phase.target_x) + abs(snapshot.y - phase.target_y)
