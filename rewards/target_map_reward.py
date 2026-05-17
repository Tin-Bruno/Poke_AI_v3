"""Reward para trocar para o mapa alvo da fase."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class TargetMapReward(RewardComponent):
    def __init__(self, success_reward: float = 10.0) -> None:
        self.success_reward = success_reward
        self.rewarded = False

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.rewarded = False

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        if phase.target_map is None or self.rewarded or snapshot.map_id != phase.target_map:
            return RewardResult(0.0, terms={"target_map": 0.0})
        self.rewarded = True
        return RewardResult(self.success_reward, True, {"target_map": self.success_reward})
