"""Recompensa geral para treino livre depois do bootstrap por phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memory.ram_map import EVENT_BITS, GameSnapshot


FREEPLAY_RAM_SHAPE = (23,)


@dataclass
class FreeplayRewardBreakdown:
    location: float = 0.0
    map: float = 0.0
    events: float = 0.0
    badges: float = 0.0
    party: float = 0.0
    survival: float = 0.0
    repeat_penalty: float = 0.0
    step_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.location
            + self.map
            + self.events
            + self.badges
            + self.party
            + self.survival
            + self.repeat_penalty
            + self.step_penalty
        )

    @property
    def progress(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.location,
                self.map,
                self.events,
                self.badges,
                self.party,
                self.survival,
            )
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "reward_location": self.location,
            "reward_map": self.map,
            "reward_events": self.events,
            "reward_badges": self.badges,
            "reward_party": self.party,
            "reward_survival": self.survival,
            "reward_repeat_penalty": self.repeat_penalty,
            "reward_step_penalty": self.step_penalty,
        }


class FreeplayReward:
    """Reward sem alvo fixo: explora, progride eventos e fortalece o time."""

    def __init__(
        self,
        location_weight: float = 0.04,
        map_weight: float = 1.0,
        event_weight: float = 0.2,
        badge_weight: float = 10.0,
        party_weight: float = 0.05,
        survival_weight: float = 0.03,
        repeat_penalty: float = 0.002,
        step_penalty: float = 0.001,
    ) -> None:
        self.location_weight = location_weight
        self.map_weight = map_weight
        self.event_weight = event_weight
        self.badge_weight = badge_weight
        self.party_weight = party_weight
        self.survival_weight = survival_weight
        self.repeat_penalty = repeat_penalty
        self.step_penalty = step_penalty
        self.seen_locations: set[tuple[int, int, int]] = set()
        self.seen_maps: set[int] = set()
        self.best_levels_by_species: dict[int, list[int]] = {}
        self.best_known_party_level_total = 0
        self.best_snapshot: GameSnapshot | None = None
        self.previous_snapshot: GameSnapshot | None = None

    def reset(self, snapshot: GameSnapshot) -> None:
        self.seen_locations = {coarse_location_key(snapshot)}
        self.seen_maps = {snapshot.map_id}
        self.best_levels_by_species = species_level_lists(snapshot)
        self.best_known_party_level_total = total_known_levels(self.best_levels_by_species)
        self.best_snapshot = snapshot
        self.previous_snapshot = snapshot

    def score(self, snapshot: GameSnapshot) -> tuple[float, bool, dict[str, float]]:
        if self.previous_snapshot is None or self.best_snapshot is None:
            self.reset(snapshot)
            return 0.0, False, FreeplayRewardBreakdown().as_dict()

        previous = self.previous_snapshot
        best = self.best_snapshot
        breakdown = FreeplayRewardBreakdown(step_penalty=-self.step_penalty)

        location_key = coarse_location_key(snapshot)
        if location_key not in self.seen_locations:
            self.seen_locations.add(location_key)
            breakdown.location += self.location_weight
        else:
            breakdown.repeat_penalty -= self.repeat_penalty

        if snapshot.map_id not in self.seen_maps:
            self.seen_maps.add(snapshot.map_id)
            breakdown.map += self.map_weight

        event_delta = max(0, snapshot.event_count - best.event_count)
        badge_delta = max(0, snapshot.badge_count - best.badge_count)
        known_party_level_total = self._update_known_party_levels(snapshot)
        party_delta = max(0, known_party_level_total - self.best_known_party_level_total)
        hp_delta = max(0.0, snapshot.hp_fraction - previous.hp_fraction)

        breakdown.events += event_delta * self.event_weight
        breakdown.badges += badge_delta * self.badge_weight
        breakdown.party += party_delta * self.party_weight
        breakdown.survival += hp_delta * self.survival_weight

        if party_delta > 0:
            self.best_known_party_level_total = known_party_level_total
        if self._is_better(snapshot, best):
            self.best_snapshot = snapshot
        self.previous_snapshot = snapshot

        info = breakdown.as_dict()
        info.update(
            {
                "seen_locations": float(len(self.seen_locations)),
                "seen_maps": float(len(self.seen_maps)),
                "badges": float(snapshot.badge_count),
                "events": float(snapshot.event_count),
                "party_count": float(snapshot.party_count),
                "party_level_total": float(sum(snapshot.party_levels)),
                "known_party_level_total": float(self.best_known_party_level_total),
                "hp_fraction": float(snapshot.hp_fraction),
                "freeplay_progress": 1.0 if breakdown.progress else 0.0,
            }
        )
        return float(breakdown.total), breakdown.progress, info

    def _is_better(self, snapshot: GameSnapshot, best: GameSnapshot) -> bool:
        current = (snapshot.badge_count, snapshot.event_count)
        previous = (best.badge_count, best.event_count)
        return current > previous

    def _update_known_party_levels(self, snapshot: GameSnapshot) -> int:
        current_levels = species_level_lists(snapshot)
        for species, levels in current_levels.items():
            best_levels = self.best_levels_by_species.setdefault(species, [])
            for index, level in enumerate(levels):
                if index >= len(best_levels):
                    best_levels.append(level)
                elif level > best_levels[index]:
                    best_levels[index] = level
            best_levels.sort(reverse=True)
        return total_known_levels(self.best_levels_by_species)


def coarse_location_key(snapshot: GameSnapshot) -> tuple[int, int, int]:
    return snapshot.map_id, snapshot.x // 2, snapshot.y // 2


def species_level_lists(snapshot: GameSnapshot) -> dict[int, list[int]]:
    levels_by_species: dict[int, list[int]] = {}
    for species, level in zip(snapshot.party_species, snapshot.party_levels):
        if species <= 0 or level <= 0:
            continue
        levels_by_species.setdefault(species, []).append(level)

    for levels in levels_by_species.values():
        levels.sort(reverse=True)
    return levels_by_species


def total_known_levels(levels_by_species: dict[int, list[int]]) -> int:
    return sum(sum(levels) for levels in levels_by_species.values())


def snapshot_vector(snapshot: GameSnapshot) -> np.ndarray:
    levels = list(snapshot.party_levels[:6]) + [0] * max(0, 6 - len(snapshot.party_levels))
    species = list(snapshot.party_species[:6]) + [0] * max(0, 6 - len(snapshot.party_species))
    event_density = snapshot.event_count / max(1, EVENT_BITS)
    values = [
        snapshot.map_id / 255.0,
        snapshot.x / 255.0,
        snapshot.y / 255.0,
        snapshot.badge_count / 8.0,
        snapshot.party_count / 6.0,
        sum(snapshot.party_levels) / 600.0,
        snapshot.max_party_level / 100.0,
        snapshot.hp_fraction,
        1.0 if snapshot.in_battle else 0.0,
        event_density,
        snapshot.max_opponent_level / 100.0,
        *[level / 100.0 for level in levels[:6]],
        *[value / 255.0 for value in species[:6]],
    ]
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
