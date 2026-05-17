"""Ambiente Gymnasium para Pokemon Red/Blue no PyBoy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from poke_ai_v3.actions import ACTIONS, action_name
from poke_ai_v3.memory import EVENT_BITS, GameSnapshot, PokemonRedMemoryReader
from poke_ai_v3.rewards import ProgressReward, RewardConfig

Observation = np.ndarray | dict[str, np.ndarray]


class PokemonRedEnv(gym.Env):
    """Ambiente de RL que controla Pokemon Red/Blue por botoes de Game Boy."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | Path,
        state_path: str | Path | None = None,
        symbols_path: str | Path | None = None,
        window: str = "null",
        action_frames: int = 12,
        warmup_frames: int = 120,
        max_steps: int | None = None,
        observation_mode: str = "screen",
        frame_stacks: int = 3,
        coords_pad: int = 12,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.rom_path = Path(rom_path)
        self.state_path = Path(state_path) if state_path else None
        self.symbols_path = Path(symbols_path) if symbols_path else None
        self.window = window
        self.action_frames = action_frames
        self.warmup_frames = warmup_frames
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.frame_stacks = frame_stacks
        self.coords_pad = coords_pad
        self.enc_freqs = 8
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = self._build_observation_space()

        self._pyboy: Any | None = None
        self._memory: PokemonRedMemoryReader | None = None
        self._reward = ProgressReward(reward_config)
        self._step_count = 0
        self._seen_coords: dict[tuple[int, int, int], int] = {}
        self._recent_screens = np.zeros((self.frame_stacks, 72, 80), dtype=np.uint8)
        self._recent_actions = np.zeros((self.frame_stacks,), dtype=np.int64)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._restart_emulator()
        self._step_count = 0

        if self.warmup_frames > 0:
            self._pyboy.tick(self.warmup_frames, True, False)

        snapshot = self._snapshot()
        self._seen_coords = {}
        self._recent_screens = np.zeros((self.frame_stacks, 72, 80), dtype=np.uint8)
        self._recent_actions = np.zeros((self.frame_stacks,), dtype=np.int64)
        self._record_position(snapshot)
        self._reward.reset(snapshot)
        return self._observation(snapshot), {"snapshot": snapshot.to_dict()}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        button = action_name(action)
        if button != "noop":
            self._pyboy.button(button, self.action_frames)

        running = self._pyboy.tick(self.action_frames, True, False)
        snapshot = self._snapshot()
        self._step_count += 1
        self._record_position(snapshot)
        self._update_recent_actions(action)
        reward, terminated, truncated, reward_info = self._reward.step(snapshot)

        info: dict[str, Any] = {
            "action": button,
            "step_count": self._step_count,
            "seen_coords": len(self._seen_coords),
            "snapshot": snapshot.to_dict(),
            **reward_info,
        }
        if self.max_steps is not None and self._step_count >= self.max_steps:
            truncated = True
        if not running:
            terminated = True

        return self._observation(snapshot), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return np.asarray(self._pyboy.screen.ndarray[:, :, :3]).copy()

    def close(self) -> None:
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None
            self._memory = None

    def _restart_emulator(self) -> None:
        self.close()
        if not self.rom_path.exists():
            raise FileNotFoundError(f"ROM nao encontrada: {self.rom_path}")
        if self.state_path and not self.state_path.exists():
            raise FileNotFoundError(f"Estado nao encontrado: {self.state_path}")
        if self.symbols_path and not self.symbols_path.exists():
            raise FileNotFoundError(f"Arquivo de simbolos nao encontrado: {self.symbols_path}")

        try:
            from pyboy import PyBoy
        except ImportError as exc:
            raise RuntimeError("Instale as dependencias com: python -m pip install -e .") from exc

        kwargs: dict[str, Any] = {
            "window": self.window,
            "sound_emulated": False,
        }
        if self.symbols_path:
            kwargs["symbols"] = str(self.symbols_path)

        self._pyboy = PyBoy(str(self.rom_path), **kwargs)
        self._pyboy.set_emulation_speed(0)

        if self.state_path:
            with self.state_path.open("rb") as state_file:
                self._pyboy.load_state(state_file)

        self._memory = PokemonRedMemoryReader(self._pyboy)

    def _snapshot(self) -> GameSnapshot:
        if self._memory is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")
        return self._memory.snapshot()

    def _build_observation_space(self) -> spaces.Space:
        if self.observation_mode == "screen":
            return spaces.Box(low=0, high=255, shape=(1, 72, 80), dtype=np.uint8)

        if self.observation_mode == "multi":
            return spaces.Dict(
                {
                    "screens": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.frame_stacks, 72, 80),
                        dtype=np.uint8,
                    ),
                    "health": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "level": spaces.Box(low=-1.0, high=1.0, shape=(self.enc_freqs,), dtype=np.float32),
                    "badges": spaces.MultiBinary(8),
                    "events": spaces.MultiBinary(EVENT_BITS),
                    "map": spaces.Box(
                        low=0,
                        high=255,
                        shape=(1, self.coords_pad * 4, self.coords_pad * 4),
                        dtype=np.uint8,
                    ),
                    "recent_actions": spaces.MultiDiscrete([len(ACTIONS)] * self.frame_stacks),
                }
            )

        raise ValueError("observation_mode deve ser 'screen' ou 'multi'")

    def _observation(self, snapshot: GameSnapshot) -> Observation:
        screen = self._screen_observation()[0]
        if self.observation_mode == "screen":
            return screen[np.newaxis, :, :]

        self._update_recent_screens(screen)
        return {
            "screens": self._recent_screens.copy(),
            "health": np.array([snapshot.hp_fraction], dtype=np.float32),
            "level": self._fourier_encode(0.02 * sum(snapshot.party_levels)),
            "badges": np.array([int(bit) for bit in f"{snapshot.badges:08b}"], dtype=np.int8),
            "events": np.array(snapshot.event_bits, dtype=np.int8),
            "map": self._local_explore_map(snapshot)[np.newaxis, :, :],
            "recent_actions": self._recent_actions.copy(),
        }

    def _screen_observation(self) -> np.ndarray:
        screen = np.asarray(self._pyboy.screen.ndarray[:, :, :3])
        grayscale = screen.mean(axis=2).astype(np.uint8)
        downsampled = grayscale[::2, ::2]
        return downsampled[np.newaxis, :, :].copy()

    def _record_position(self, snapshot: GameSnapshot) -> None:
        if snapshot.in_battle:
            return
        key = self._position_key(snapshot)
        self._seen_coords[key] = self._seen_coords.get(key, 0) + 1

    def _local_explore_map(self, snapshot: GameSnapshot) -> np.ndarray:
        center_y = snapshot.y
        center_x = snapshot.x
        base = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)

        for map_id, y, x in self._seen_coords:
            if map_id != snapshot.map_id:
                continue
            row = y - center_y + self.coords_pad
            col = x - center_x + self.coords_pad
            if 0 <= row < base.shape[0] and 0 <= col < base.shape[1]:
                base[row, col] = 255

        return np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)

    def _update_recent_screens(self, screen: np.ndarray) -> None:
        self._recent_screens = np.roll(self._recent_screens, 1, axis=0)
        self._recent_screens[0] = screen

    def _update_recent_actions(self, action: int) -> None:
        self._recent_actions = np.roll(self._recent_actions, 1)
        self._recent_actions[0] = int(action)

    def _fourier_encode(self, value: float) -> np.ndarray:
        return np.sin(value * 2 ** np.arange(self.enc_freqs)).astype(np.float32)

    @staticmethod
    def _position_key(snapshot: GameSnapshot) -> tuple[int, int, int]:
        return snapshot.map_id, snapshot.y, snapshot.x
