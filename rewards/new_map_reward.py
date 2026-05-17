"""Reward de exploracao por mapas e coordenadas novas."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class NewMapReward(RewardComponent):
    def __init__(self, coord_reward: float = 0.02, map_reward: float = 1.0) -> None:
        self.coord_reward = coord_reward
        self.map_reward = map_reward
        self.seen_positions: set[tuple[int, int, int]] = set()
        self.seen_maps: set[int] = set()

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.seen_positions = {snapshot.position}
        self.seen_maps = {snapshot.map_id}

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        if snapshot.in_battle:
            return RewardResult(0.0, terms={"new_map": 0.0})

        reward = 0.0
        progress = False
        if snapshot.position not in self.seen_positions:
            self.seen_positions.add(snapshot.position)
            reward += self.coord_reward
            progress = True
        if snapshot.map_id not in self.seen_maps:
            self.seen_maps.add(snapshot.map_id)
            reward += self.map_reward
            progress = True
        return RewardResult(reward, progress, {"new_map": reward})
