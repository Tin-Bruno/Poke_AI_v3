"""Reward basica para fases de batalha."""

from __future__ import annotations

from memory.ram_map import GameSnapshot
from phases.phase_config import PhaseConfig
from rewards.base import RewardComponent, RewardResult


class BattleReward(RewardComponent):
    def __init__(
        self,
        win_reward: float = 15.0,
        level_reward: float = 3.0,
        enter_battle_reward: float = 1.0,
        alive_reward: float = 0.01,
        hp_penalty: float = 0.0,
        faint_penalty: float = 8.0,
    ) -> None:
        self.win_reward = win_reward
        self.level_reward = level_reward
        self.enter_battle_reward = enter_battle_reward
        self.alive_reward = alive_reward
        self.hp_penalty = hp_penalty
        self.faint_penalty = faint_penalty

        self.started_in_battle = False
        self.was_in_battle = False
        self.best_level = 0
        self.last_hp_fraction = 0.0
        self.win_rewarded = False
        self.enter_rewarded = False
        self.faint_penalized = False

    def reset(self, snapshot: GameSnapshot, phase: PhaseConfig) -> None:
        self.started_in_battle = snapshot.in_battle
        self.was_in_battle = snapshot.in_battle
        self.best_level = snapshot.max_party_level
        self.last_hp_fraction = snapshot.hp_fraction
        self.win_rewarded = False
        self.enter_rewarded = False
        self.faint_penalized = False

    def step(self, snapshot: GameSnapshot, phase: PhaseConfig) -> RewardResult:
        reward = 0.0
        progress = False

        # Pequena reward por estar em batalha, para não tratar batalha como estado ruim.
        if snapshot.in_battle and not self.enter_rewarded:
            reward += self.enter_battle_reward
            self.enter_rewarded = True
            progress = True

        # Enquanto ainda está em batalha e vivo, dá um incentivo pequeno.
        if snapshot.in_battle and snapshot.hp_fraction > 0:
            reward += self.alive_reward

        # Level up é bom, mas não deve ser a única fonte de reward.
        if snapshot.max_party_level > self.best_level:
            gained = snapshot.max_party_level - self.best_level
            reward += gained * self.level_reward
            self.best_level = snapshot.max_party_level
            progress = True

        # Para a primeira batalha, deixe hp_penalty = 0.0.
        hp_loss = max(self.last_hp_fraction - snapshot.hp_fraction, 0.0)
        reward -= hp_loss * self.hp_penalty
        self.last_hp_fraction = snapshot.hp_fraction

        # Vitória: começou em batalha, saiu da batalha e ainda está vivo.
        if (
            self.started_in_battle
            and self.was_in_battle
            and not snapshot.in_battle
            and snapshot.party_count > 0
            and snapshot.hp_fraction > 0
            and not self.win_rewarded
        ):
            reward += self.win_reward
            self.win_rewarded = True
            progress = True

        # Derrota/faint.
        if (
            self.started_in_battle
            and snapshot.party_count > 0
            and snapshot.hp_fraction <= 0
            and not self.faint_penalized
        ):
            reward -= self.faint_penalty
            self.faint_penalized = True

        self.was_in_battle = snapshot.in_battle

        return RewardResult(reward, progress, {"battle": reward})