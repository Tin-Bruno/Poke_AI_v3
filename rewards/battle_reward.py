"""Reward basica para fases de batalha."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class BattleReward(RewardComponent):
    def __init__(self, level_reward: float = 1.0, hp_penalty: float = 0.2) -> None:
        self.level_reward = level_reward
        self.hp_penalty = hp_penalty
        self.best_level = 0
        self.last_hp_fraction = 0.0

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.best_level = snapshot.max_party_level
        self.last_hp_fraction = snapshot.hp_fraction

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        reward = 0.0
        progress = False

        if snapshot.max_party_level > self.best_level:
            gained = snapshot.max_party_level - self.best_level
            reward += gained * self.level_reward
            self.best_level = snapshot.max_party_level
            progress = True

        hp_loss = max(self.last_hp_fraction - snapshot.hp_fraction, 0.0)
        reward -= hp_loss * self.hp_penalty
        self.last_hp_fraction = snapshot.hp_fraction

        return RewardResult(reward, progress, {"battle": reward})
