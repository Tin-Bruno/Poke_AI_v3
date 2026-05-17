"""Composicao de rewards declaradas em PhaseConfig."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent
from rewards.battle_reward import BattleReward
from rewards.dialog_reward import DialogReward
from rewards.new_map_reward import NewMapReward
from rewards.target_map_reward import TargetMapReward
from rewards.target_position_reward import TargetPositionReward


REWARD_REGISTRY: dict[str, type[RewardComponent]] = {
    "target_position": TargetPositionReward,
    "target_map": TargetMapReward,
    "new_map": NewMapReward,
    "dialog": DialogReward,
    "battle": BattleReward,
}


class PhaseReward:
    def __init__(
        self,
        phase: PhaseConfig,
        step_penalty: float = -0.005,
        scale: float = 1.0,
        max_no_progress_steps: int = 200,
    ) -> None:
        self.phase = phase
        self.step_penalty = step_penalty
        self.scale = scale
        self.max_no_progress_steps = max_no_progress_steps
        self.components = [REWARD_REGISTRY[name]() for name in phase.rewards]
        self.steps_since_progress = 0

    def reset(self, snapshot: GameSnapshot) -> None:
        for component in self.components:
            component.reset(snapshot, self.phase)
        self.steps_since_progress = 0

    def step(self, snapshot: GameSnapshot) -> tuple[float, bool, dict[str, float | int]]:
        reward = self.step_penalty
        progress = False
        terms: dict[str, float | int] = {"step_penalty": self.step_penalty}

        for component in self.components:
            result = component.step(snapshot, self.phase)
            reward += result.value
            progress = progress or result.progress
            if result.terms:
                terms.update(result.terms)

        if progress:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        terms["steps_since_progress"] = self.steps_since_progress
        scaled = reward * self.scale
        terms["reward_unscaled"] = reward
        terms["reward_scaled"] = scaled
        return scaled, self.steps_since_progress >= self.max_no_progress_steps, terms


def make_reward(
    phase: PhaseConfig,
    step_penalty: float = -0.005,
    scale: float = 1.0,
    max_no_progress_steps: int = 200,
) -> PhaseReward:
    return PhaseReward(
        phase=phase,
        step_penalty=step_penalty,
        scale=scale,
        max_no_progress_steps=max_no_progress_steps,
    )
