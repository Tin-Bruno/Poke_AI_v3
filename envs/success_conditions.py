"""Condicoes de sucesso independentes do ambiente."""

from __future__ import annotations

from typing import Callable

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig


SuccessFn = Callable[[GameSnapshot, GameSnapshot, PhaseConfig], bool]


def target_map(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return phase.target_map is not None and snapshot.map_id == phase.target_map


def target_position(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    if phase.target_map is not None and snapshot.map_id != phase.target_map:
        return False
    if phase.target_x is None or phase.target_y is None:
        return False
    distance = abs(snapshot.x - phase.target_x) + abs(snapshot.y - phase.target_y)
    return distance <= phase.target_radius


def dialog_or_map_change(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return snapshot.event_count > initial.event_count or snapshot.map_id != initial.map_id


def event_count_increase(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return snapshot.event_count > initial.event_count


def badge_count(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return phase.target_badges is not None and snapshot.badge_count >= phase.target_badges


SUCCESS_CONDITIONS: dict[str, SuccessFn] = {
    "target_map": target_map,
    "target_position": target_position,
    "dialog_or_map_change": dialog_or_map_change,
    "event_count_increase": event_count_increase,
    "badge_count": badge_count,
}


class SuccessChecker:
    def __init__(self, phase: PhaseConfig, initial: GameSnapshot) -> None:
        self.phase = phase
        self.initial = initial
        self.fn = SUCCESS_CONDITIONS[phase.success]

    def check(self, snapshot: GameSnapshot) -> bool:
        return self.fn(snapshot, self.initial, self.phase)
