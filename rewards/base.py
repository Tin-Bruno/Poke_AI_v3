"""Interfaces pequenas para rewards por fase."""

from __future__ import annotations

from dataclasses import dataclass

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig


@dataclass
class RewardResult:
    value: float
    progress: bool = False
    terms: dict[str, float] | None = None


class RewardComponent:
    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        pass

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        raise NotImplementedError
