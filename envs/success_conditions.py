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


def target_position_after_event_count(
    snapshot: GameSnapshot,
    initial: GameSnapshot,
    phase: PhaseConfig,
) -> bool:
    required_delta = max(1, phase.target_event_count_delta)
    has_event_progress = snapshot.event_count >= initial.event_count + required_delta
    return has_event_progress and target_position(snapshot, initial, phase)


def dialog_or_map_change(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return snapshot.event_count > initial.event_count or snapshot.map_id != initial.map_id


def event_count_increase(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return snapshot.event_count > initial.event_count


def party_count_increase(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return snapshot.party_count > initial.party_count


def battle_started(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return not initial.in_battle and snapshot.in_battle


def badge_count(snapshot: GameSnapshot, initial: GameSnapshot, phase: PhaseConfig) -> bool:
    return phase.target_badges is not None and snapshot.badge_count >= phase.target_badges


SUCCESS_CONDITIONS: dict[str, SuccessFn] = {
    "target_map": target_map,
    "target_position": target_position,
    "target_position_after_event_count": target_position_after_event_count,
    "dialog_or_map_change": dialog_or_map_change,
    "event_count_increase": event_count_increase,
    "party_count_increase": party_count_increase,
    "battle_started": battle_started,
    "badge_count": badge_count,
}


class SuccessChecker:
    def __init__(self, phase: PhaseConfig, initial: GameSnapshot) -> None:
        self.phase = phase
        self.initial = initial
        self.fn = SUCCESS_CONDITIONS[phase.success]

    def check(self, snapshot: GameSnapshot) -> bool:
        return self.fn(snapshot, self.initial, self.phase)
