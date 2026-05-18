"""Reward para ganhar um Pokemon no time."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class PartyReward(RewardComponent):
    def __init__(self, gained_reward: float = 10.0) -> None:
        self.gained_reward = gained_reward
        self.best_party_count = 0

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.best_party_count = snapshot.party_count

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        if snapshot.party_count <= self.best_party_count:
            return RewardResult(0.0, terms={"party": 0.0})

        gained = snapshot.party_count - self.best_party_count
        self.best_party_count = snapshot.party_count
        reward = gained * self.gained_reward
        return RewardResult(reward, True, {"party": reward})
