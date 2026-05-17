"""Reward para eventos e progresso de dialogo."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class DialogReward(RewardComponent):
    def __init__(self, event_reward: float = 2.0) -> None:
        self.event_reward = event_reward
        self.best_event_count = 0

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.best_event_count = snapshot.event_count

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        if snapshot.event_count <= self.best_event_count:
            return RewardResult(0.0, terms={"dialog": 0.0})

        gained = snapshot.event_count - self.best_event_count
        self.best_event_count = snapshot.event_count
        reward = gained * self.event_reward
        return RewardResult(reward, True, {"dialog": reward})
