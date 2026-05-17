"""Reward shaping para transformar progresso de Pokemon em sinal de RL."""

from __future__ import annotations

from dataclasses import dataclass

from poke_ai_v3.memory import GameSnapshot


@dataclass
class RewardConfig:
    step_penalty: float = -0.005
    new_position_reward: float = 0.05
    new_map_reward: float = 2.0
    event_reward: float = 4.0
    heal_reward: float = 10.0
    party_count_reward: float = 5.0
    level_reward: float = 1.0
    badge_reward: float = 50.0
    opponent_level_reward: float = 0.2
    repeat_position_threshold: int = 600
    repeat_position_penalty: float = -0.05
    explore_weight: float = 1.0
    reward_scale: float = 1.0
    complete_badges: int = 8
    max_no_progress_steps: int = 2000


class ProgressReward:
    """Calcula recompensa densa baseada em exploracao e marcos permanentes."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.visited_positions: set[tuple[int, int, int]] = set()
        self.position_visit_counts: dict[tuple[int, int, int], int] = {}
        self.visited_maps: set[int] = set()
        self.best_badges = 0
        self.best_event_count = 0
        self.best_party_count = 0
        self.best_max_level = 0
        self.best_opponent_level = 0
        self.last_hp_fraction = 0.0
        self.steps_since_progress = 0

    def reset(self, snapshot: GameSnapshot) -> None:
        position = self._position_key(snapshot)
        self.visited_positions = {position}
        self.position_visit_counts = {position: 1}
        self.visited_maps = {snapshot.map_id}
        self.best_badges = snapshot.badge_count
        self.best_event_count = snapshot.event_count
        self.best_party_count = snapshot.party_count
        self.best_max_level = snapshot.max_party_level
        self.best_opponent_level = snapshot.max_opponent_level
        self.last_hp_fraction = snapshot.hp_fraction
        self.steps_since_progress = 0

    def step(self, snapshot: GameSnapshot) -> tuple[float, bool, bool, dict[str, int | float]]:
        reward = self.config.step_penalty
        progress_events = 0

        position = self._position_key(snapshot)
        self.position_visit_counts[position] = self.position_visit_counts.get(position, 0) + 1

        if not snapshot.in_battle and position not in self.visited_positions:
            self.visited_positions.add(position)
            reward += self.config.new_position_reward * self.config.explore_weight
            progress_events += 1

        if not snapshot.in_battle and snapshot.map_id not in self.visited_maps:
            self.visited_maps.add(snapshot.map_id)
            reward += self.config.new_map_reward * self.config.explore_weight
            progress_events += 1

        if self.position_visit_counts[position] > self.config.repeat_position_threshold:
            reward += self.config.repeat_position_penalty

        if snapshot.event_count > self.best_event_count:
            gained = snapshot.event_count - self.best_event_count
            reward += gained * self.config.event_reward
            self.best_event_count = snapshot.event_count
            progress_events += gained

        if snapshot.badge_count > self.best_badges:
            gained = snapshot.badge_count - self.best_badges
            reward += gained * self.config.badge_reward
            self.best_badges = snapshot.badge_count
            progress_events += gained

        if snapshot.party_count > self.best_party_count:
            gained = snapshot.party_count - self.best_party_count
            reward += gained * self.config.party_count_reward
            self.best_party_count = snapshot.party_count
            progress_events += gained

        if snapshot.max_party_level > self.best_max_level:
            gained = snapshot.max_party_level - self.best_max_level
            reward += gained * self.config.level_reward
            self.best_max_level = snapshot.max_party_level
            progress_events += gained

        if snapshot.max_opponent_level > self.best_opponent_level:
            gained = snapshot.max_opponent_level - self.best_opponent_level
            reward += gained * self.config.opponent_level_reward
            self.best_opponent_level = snapshot.max_opponent_level
            progress_events += gained

        if snapshot.hp_fraction > self.last_hp_fraction and snapshot.party_count <= self.best_party_count:
            healed = snapshot.hp_fraction - self.last_hp_fraction
            reward += (healed * healed) * self.config.heal_reward

        self.last_hp_fraction = snapshot.hp_fraction

        if progress_events:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        terminated = snapshot.badge_count >= self.config.complete_badges
        truncated = self.steps_since_progress >= self.config.max_no_progress_steps
        scaled_reward = reward * self.config.reward_scale
        info = {
            "reward_total": scaled_reward,
            "reward_unscaled": reward,
            "progress_events": progress_events,
            "visited_positions": len(self.visited_positions),
            "visited_maps": len(self.visited_maps),
            "badge_count": snapshot.badge_count,
            "event_count": snapshot.event_count,
            "party_count": snapshot.party_count,
            "hp_fraction": snapshot.hp_fraction,
            "max_party_level": snapshot.max_party_level,
            "max_opponent_level": snapshot.max_opponent_level,
            "steps_since_progress": self.steps_since_progress,
        }
        return scaled_reward, terminated, truncated, info

    @staticmethod
    def _position_key(snapshot: GameSnapshot) -> tuple[int, int, int]:
        return snapshot.map_id, snapshot.y, snapshot.x
