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

FREEPLAY_ACTIONS: tuple[tuple[str, ...], ...] = (
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
    emulation_speed: float | None = None


class FreeplayPokemonRedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: FreeplayConfig,
        reward_model: FreeplayReward | None = None,
        actions: tuple[tuple[str, ...], ...] = FREEPLAY_ACTIONS,
    ) -> None:
        super().__init__()
        self.config = config
        self.reward_model = reward_model or FreeplayReward()
        self.actions = actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict(
            {
                "screen": spaces.Box(low=0, high=255, shape=(1, 72, 80), dtype=np.uint8),
                "ram": spaces.Box(low=0.0, high=1.0, shape=FREEPLAY_RAM_SHAPE, dtype=np.float32),
            }
        )

        self._pyboy: Any | None = None
        self._reader: PokemonRedRamReader | None = None
        self._step_count = 0
        self._steps_since_progress = 0
        self._best_seen_locations = 0
        self._last_action = "noop"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self._restart_emulator()
        self._step_count = 0
        self._steps_since_progress = 0
        self._best_seen_locations = 0
        self._last_action = "noop"

        if self.config.warmup_frames > 0:
            self._tick_many(self.config.warmup_frames)

        snapshot = self._snapshot()
        self.reward_model.reset(snapshot)
        return self._observation(snapshot), self._info(snapshot, reward=0.0)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._pyboy is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")

        self._step_count += 1
        combo = self.actions[int(action)]
        self._last_action = "+".join(combo) if combo else "noop"
        running = self._run_combo(combo)

        snapshot = self._snapshot()
        reward, progress, reward_info = self.reward_model.score(snapshot)
        seen_locations = int(reward_info.get("seen_locations", 0))
        if progress or seen_locations > self._best_seen_locations:
            self._best_seen_locations = max(self._best_seen_locations, seen_locations)
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

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

    def _observation(self, snapshot: GameSnapshot) -> dict[str, np.ndarray]:
        screen = np.asarray(self._pyboy.screen.ndarray[:, :, :3])
        grayscale = screen.mean(axis=2).astype(np.uint8)
        small = grayscale[::2, ::2][np.newaxis, :, :].copy()
        return {"screen": small, "ram": snapshot_vector(snapshot)}

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
            "action": self._last_action,
        }
