"""Callbacks leves para acompanhar treino no TensorBoard."""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class PokeStatsCallback(BaseCallback):
    """Registra metricas do ambiente vindas do dicionario info."""

    TRACKED_KEYS = (
        "visited_positions",
        "visited_maps",
        "event_count",
        "badge_count",
        "party_count",
        "hp_fraction",
        "max_party_level",
        "max_opponent_level",
        "steps_since_progress",
        "reward_unscaled",
    )

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            for key in self.TRACKED_KEYS:
                if key in info:
                    self.logger.record(f"env/{key}", info[key])
        return True
