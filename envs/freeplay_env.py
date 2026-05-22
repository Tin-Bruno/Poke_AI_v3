"""Ambiente livre para continuar o jogo sem curriculo por fases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from memory.ram_map import GameSnapshot, PokemonRedRamReader
from rewards.freeplay_reward import FREEPLAY_RAM_SHAPE, FreeplayReward, snapshot_vector

FREEPLAY_SIMPLE_ACTIONS: tuple[tuple[str, ...], ...] = (
    (),
    ("up",),
    ("down",),
    ("left",),
    ("right",),
    ("a",),
    ("b",),
)

FREEPLAY_COMBO_ACTIONS: tuple[tuple[str, ...], ...] = (
    (),
    ("up",),
    ("down",),
    ("left",),
    ("right",),
    ("a",),
    ("b",),
    ("start",),
    ("select",),
    ("up", "a"),
    ("down", "a"),
    ("left", "a"),
    ("right", "a"),
    ("up", "b"),
    ("down", "b"),
    ("left", "b"),
    ("right", "b"),
)

FREEPLAY_ACTIONS = FREEPLAY_COMBO_ACTIONS
FREEPLAY_ACTION_SETS = {
    "simple": FREEPLAY_SIMPLE_ACTIONS,
    "combo": FREEPLAY_COMBO_ACTIONS,
}
RECENT_ACTIONS = 3
MOVEMENT_BUTTONS = {"up", "down", "left", "right"}


@dataclass(frozen=True)
class FreeplayConfig:
    rom_path: str | Path
    state_path: str | Path | None = "states/phase9_start.state"
    symbols_path: str | Path | None = None
    window: str = "null"
    action_frames: int = 24
    release_frames: int = 6
    warmup_frames: int = 0
    max_steps: int = 20_000
    stagnation_steps: int = 4_000
    action_set: str = "simple"
    observation_mode: str = "multi"
    memory_observation: bool = True
    visited_radius: int = 12
    blocked_move_penalty: float = 0.02
    emulation_speed: float | None = None


class FreeplayPokemonRedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: FreeplayConfig,
        reward_model: FreeplayReward | None = None,
        actions: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.reward_model = reward_model or FreeplayReward()
        self.actions = actions or self._actions_from_config(config.action_set)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = self._build_observation_space()

        self._pyboy: Any | None = None
        self._reader: PokemonRedRamReader | None = None
        self._step_count = 0
        self._steps_since_progress = 0
        self._best_seen_locations = 0
        self._best_seen_maps = 0
        self._best_event_count = 0
        self._best_party_level_total = 0
        self._battle_count = 0
        self._battle_win_count = 0
        self._was_in_battle = False
        self._visited_coords: set[tuple[int, int, int]] = set()
        self._recent_actions: list[int | None] = [None] * RECENT_ACTIONS
        self._last_action = "noop"
        self._last_blocked_penalty = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self._restart_emulator()
        self._step_count = 0
        self._steps_since_progress = 0
        self._best_seen_locations = 0
        self._best_seen_maps = 0
        self._best_event_count = 0
        self._best_party_level_total = 0
        self._battle_count = 0
        self._battle_win_count = 0
        self._was_in_battle = False
        self._visited_coords = set()
        self._recent_actions = [None] * RECENT_ACTIONS
        self._last_action = "noop"
        self._last_blocked_penalty = 0.0

        if self.config.warmup_frames > 0:
            self._tick_many(self.config.warmup_frames)

        snapshot = self._snapshot()
        self.reward_model.reset(snapshot)
        self._best_event_count = snapshot.event_count
        self._best_party_level_total = sum(snapshot.party_levels)
        self._best_seen_maps = 1
        self._best_seen_locations = 1
        self._was_in_battle = snapshot.in_battle
        self._remember_position(snapshot)
        return self._observation(snapshot), self._info(snapshot, reward=0.0)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._pyboy is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")

        self._step_count += 1
        combo = self.actions[int(action)]
        self._last_action = "+".join(combo) if combo else "noop"
        previous_snapshot = self._snapshot()
        running = self._run_combo(combo)

        snapshot = self._snapshot()
        reward, progress, reward_info = self.reward_model.score(snapshot)
        self._update_progress_metrics(previous_snapshot, snapshot)
        blocked_penalty = self._blocked_move_penalty(previous_snapshot, snapshot, combo)
        if blocked_penalty:
            reward += blocked_penalty
            reward_info["reward_blocked_move_penalty"] = blocked_penalty
        self._last_blocked_penalty = blocked_penalty
        self._remember_position(snapshot)
        self._remember_action(int(action))
        seen_locations = int(reward_info.get("seen_locations", 0))
        seen_maps = int(reward_info.get("seen_maps", 0))
        if progress or seen_locations > self._best_seen_locations:
            self._best_seen_locations = max(self._best_seen_locations, seen_locations)
            self._steps_since_progress = 0
        elif seen_maps > self._best_seen_maps:
            self._best_seen_maps = seen_maps
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        self._best_seen_maps = max(self._best_seen_maps, seen_maps)

        terminated = not running
        truncated = self._step_count >= self.config.max_steps
        if self.config.stagnation_steps > 0:
            truncated = truncated or self._steps_since_progress >= self.config.stagnation_steps

        info = self._info(snapshot, reward=reward)
        info.update(reward_info)
        info["steps_since_progress"] = self._steps_since_progress
        info["action"] = self._last_action
        return self._observation(snapshot), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._pyboy is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")
        return np.asarray(self._pyboy.screen.ndarray[:, :, :3]).copy()

    def close(self) -> None:
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None
            self._reader = None

    def _restart_emulator(self) -> None:
        self.close()
        self._check_paths()

        try:
            from pyboy import PyBoy
        except ImportError as exc:
            raise RuntimeError("Instale as dependencias com: python -m pip install -r requirements.txt") from exc

        kwargs: dict[str, Any] = {"window": self.config.window}
        if self.config.window == "null":
            kwargs["sound_emulated"] = False
        if self.config.symbols_path:
            kwargs["symbols"] = str(self.config.symbols_path)

        self._pyboy = PyBoy(str(self.config.rom_path), **kwargs)
        speed = self.config.emulation_speed
        if speed is None:
            speed = 0 if self.config.window == "null" else 1
        self._pyboy.set_emulation_speed(speed)

        if self.config.state_path:
            with Path(self.config.state_path).open("rb") as state_file:
                self._pyboy.load_state(state_file)

        self._reader = PokemonRedRamReader(self._pyboy)

    def _actions_from_config(self, action_set: str) -> tuple[tuple[str, ...], ...]:
        try:
            return FREEPLAY_ACTION_SETS[action_set]
        except KeyError as exc:
            valid = ", ".join(sorted(FREEPLAY_ACTION_SETS))
            raise ValueError(f"action_set invalido: {action_set}. Use: {valid}") from exc

    def _build_observation_space(self) -> spaces.Space:
        if self.config.observation_mode == "coords":
            return spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)

        if self.config.observation_mode == "ram":
            return spaces.Box(low=0.0, high=1.0, shape=FREEPLAY_RAM_SHAPE, dtype=np.float32)

        if self.config.observation_mode != "multi":
            raise ValueError("observation_mode deve ser 'coords', 'ram' ou 'multi'")

        observation_spaces: dict[str, spaces.Space] = {
            "screen": spaces.Box(low=0, high=255, shape=(1, 72, 80), dtype=np.uint8),
            "ram": spaces.Box(low=0.0, high=1.0, shape=FREEPLAY_RAM_SHAPE, dtype=np.float32),
        }
        if self.config.memory_observation:
            visited_size = max(1, self.config.visited_radius) * 4
            observation_spaces["visited"] = spaces.Box(
                low=0,
                high=255,
                shape=(1, visited_size, visited_size),
                dtype=np.uint8,
            )
            observation_spaces["recent_actions"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.actions) * RECENT_ACTIONS,),
                dtype=np.float32,
            )
        return spaces.Dict(observation_spaces)

    def _check_paths(self) -> None:
        rom_path = Path(self.config.rom_path)
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM nao encontrada: {rom_path}")
        if self.config.state_path and not Path(self.config.state_path).exists():
            raise FileNotFoundError(f"State de freeplay nao encontrado: {self.config.state_path}")
        if self.config.symbols_path and not Path(self.config.symbols_path).exists():
            raise FileNotFoundError(f"Arquivo de simbolos nao encontrado: {self.config.symbols_path}")

    def _snapshot(self) -> GameSnapshot:
        if self._reader is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")
        return self._reader.snapshot()

    def _run_combo(self, combo: tuple[str, ...]) -> bool:
        if not combo:
            return self._tick_many(self.config.action_frames)

        for button in combo:
            self._pyboy.button_press(button)
        if not self._tick_many(self.config.action_frames):
            return False
        for button in combo:
            self._pyboy.button_release(button)
        return self._tick_many(self.config.release_frames)

    def _tick_many(self, frames: int) -> bool:
        render = self.config.window != "null"
        for _ in range(max(0, frames)):
            if not bool(self._pyboy.tick(1, render, False)):
                return False
        return True

    def _observation(self, snapshot: GameSnapshot) -> np.ndarray | dict[str, np.ndarray]:
        if self.config.observation_mode == "coords":
            return np.array([snapshot.map_id, snapshot.x, snapshot.y], dtype=np.uint8)

        if self.config.observation_mode == "ram":
            return snapshot_vector(snapshot)

        screen = np.asarray(self._pyboy.screen.ndarray[:, :, :3])
        grayscale = screen.mean(axis=2).astype(np.uint8)
        small = grayscale[::2, ::2][np.newaxis, :, :].copy()
        obs = {"screen": small, "ram": snapshot_vector(snapshot)}
        if self.config.memory_observation:
            obs["visited"] = self._visited_patch(snapshot)
            obs["recent_actions"] = self._recent_actions_vector()
        return obs

    def _update_progress_metrics(self, before: GameSnapshot, after: GameSnapshot) -> None:
        if not before.in_battle and after.in_battle:
            self._battle_count += 1
        if before.in_battle and not after.in_battle and after.party_count > 0 and after.hp_fraction > 0:
            self._battle_win_count += 1
        self._best_event_count = max(self._best_event_count, after.event_count)
        self._best_party_level_total = max(self._best_party_level_total, sum(after.party_levels))
        self._was_in_battle = after.in_battle

    def _remember_position(self, snapshot: GameSnapshot) -> None:
        self._visited_coords.add((snapshot.map_id, snapshot.x, snapshot.y))

    def _remember_action(self, action: int) -> None:
        self._recent_actions = [action, *self._recent_actions[: RECENT_ACTIONS - 1]]

    def _visited_patch(self, snapshot: GameSnapshot) -> np.ndarray:
        radius = max(1, self.config.visited_radius)
        patch = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
        for map_id, x, y in self._visited_coords:
            if map_id != snapshot.map_id:
                continue
            dx = x - snapshot.x
            dy = y - snapshot.y
            if -radius <= dx < radius and -radius <= dy < radius:
                patch[dy + radius, dx + radius] = 255
        enlarged = np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)
        return enlarged[np.newaxis, :, :]

    def _recent_actions_vector(self) -> np.ndarray:
        vector = np.zeros(len(self.actions) * RECENT_ACTIONS, dtype=np.float32)
        for index, action in enumerate(self._recent_actions):
            if action is None:
                continue
            vector[index * len(self.actions) + action] = 1.0
        return vector

    def _blocked_move_penalty(
        self,
        before: GameSnapshot,
        after: GameSnapshot,
        combo: tuple[str, ...],
    ) -> float:
        if self.config.blocked_move_penalty <= 0.0:
            return 0.0
        if before.in_battle or after.in_battle:
            return 0.0
        if not any(button in MOVEMENT_BUTTONS for button in combo):
            return 0.0
        before_pos = (before.map_id, before.x, before.y)
        after_pos = (after.map_id, after.x, after.y)
        if before_pos != after_pos:
            return 0.0
        return -float(self.config.blocked_move_penalty)

    def _info(self, snapshot: GameSnapshot, reward: float) -> dict[str, Any]:
        return {
            "mode": "freeplay",
            "step_count": self._step_count,
            "reward": reward,
            "map_id": snapshot.map_id,
            "x": snapshot.x,
            "y": snapshot.y,
            "badges": snapshot.badge_count,
            "party_count": snapshot.party_count,
            "party_levels": snapshot.party_levels,
            "hp_fraction": snapshot.hp_fraction,
            "event_count": snapshot.event_count,
            "in_battle": snapshot.in_battle,
            "seen_coords": len(self._visited_coords),
            "best_seen_maps": self._best_seen_maps,
            "best_seen_locations": self._best_seen_locations,
            "best_event_count": self._best_event_count,
            "best_party_level_total": self._best_party_level_total,
            "battles_started": self._battle_count,
            "battles_won": self._battle_win_count,
            "blocked_move_penalty": self._last_blocked_penalty,
            "action": self._last_action,
        }
