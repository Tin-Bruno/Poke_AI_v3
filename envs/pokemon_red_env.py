"""Ambiente Gymnasium generico para fases de Pokemon Red."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.step_handler import ACTIONS, StepHandler, action_name
from envs.success_conditions import SuccessChecker
from memory.ram_map import EVENT_BITS, GameSnapshot, PokemonRedRamReader
from phases.phase_config import PhaseConfig, get_phase
from rewards import PhaseReward, make_reward

Observation = np.ndarray | dict[str, np.ndarray]


class PokemonRedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str | Path,
        phase: str | PhaseConfig,
        states_dir: str | Path = "states",
        state_path: str | Path | None = None,
        symbols_path: str | Path | None = None,
        window: str = "null",
        action_frames: int = 12,
        warmup_frames: int = 0,
        observation_mode: str = "multi",
        frame_stacks: int = 3,
        reward_scale: float = 1.0,
        max_no_progress_steps: int = 200,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.rom_path = Path(rom_path)
        self.phase = get_phase(phase) if isinstance(phase, str) else phase
        self.states_dir = Path(states_dir)
        self.state_path = Path(state_path) if state_path else self.states_dir / self.phase.state
        self.symbols_path = Path(symbols_path) if symbols_path else None
        self.window = window
        self.action_frames = action_frames
        self.warmup_frames = warmup_frames
        self.observation_mode = observation_mode
        self.frame_stacks = frame_stacks
        self.coords_pad = 12
        self.enc_freqs = 8
        self.render_mode = render_mode
        self.actions = self.phase.actions or ACTIONS

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = self._build_observation_space()

        self._pyboy: Any | None = None
        self._reader: PokemonRedRamReader | None = None
        self._step_handler: StepHandler | None = None
        self._reward: PhaseReward = make_reward(
            self.phase,
            scale=reward_scale,
            max_no_progress_steps=max_no_progress_steps,
        )
        self._success: SuccessChecker | None = None
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
        self._success = SuccessChecker(self.phase, snapshot)
        self._reward.reset(snapshot)
        self._seen_coords = {}
        self._recent_screens = np.zeros((self.frame_stacks, 72, 80), dtype=np.uint8)
        self._recent_actions = np.zeros((self.frame_stacks,), dtype=np.int64)
        self._record_position(snapshot)
        return self._observation(snapshot), self._info(snapshot, reward=0.0, success=False)

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._step_handler is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")

        previous_snapshot = self._snapshot()
        running = self._step_handler.run(action)
        self._step_count += 1

        snapshot = self._snapshot()
        self._record_position(snapshot)
        self._update_recent_actions(action)

        reward, no_progress_timeout, reward_terms = self._reward.step(snapshot)
        blocked_move = self._is_blocked_move(action, previous_snapshot, snapshot)
        if blocked_move and self.phase.blocked_move_penalty:
            reward += self.phase.blocked_move_penalty
            reward_terms["blocked_move_penalty"] = self.phase.blocked_move_penalty
            reward_terms["reward_scaled"] = reward
            if self._reward.scale:
                reward_terms["reward_unscaled"] = reward / self._reward.scale

        success = self._success.check(snapshot) if self._success else False
        terminated = success or not running
        truncated = self._step_count >= self.phase.max_steps or no_progress_timeout

        info = self._info(snapshot, reward=reward, success=success)
        info.update(reward_terms)
        info["action"] = action_name(action, self.actions)
        info["blocked_move"] = blocked_move
        info["no_progress_timeout"] = no_progress_timeout

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
            self._step_handler = None

    def save_state(self, path: str | Path) -> None:
        if self._pyboy is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("wb") as state_file:
            self._pyboy.save_state(state_file)

    def _restart_emulator(self) -> None:
        self.close()
        self._check_paths()

        try:
            from pyboy import PyBoy
        except ImportError as exc:
            raise RuntimeError("Instale as dependencias com: python -m pip install -r requirements.txt") from exc

        kwargs: dict[str, Any] = {"window": self.window}
        if self.window == "null":
            kwargs["sound_emulated"] = False
        if self.symbols_path:
            kwargs["symbols"] = str(self.symbols_path)

        self._pyboy = PyBoy(str(self.rom_path), **kwargs)
        self._pyboy.set_emulation_speed(0 if self.window == "null" else 1)

        with self.state_path.open("rb") as state_file:
            self._pyboy.load_state(state_file)

        self._reader = PokemonRedRamReader(self._pyboy)
        self._step_handler = StepHandler(
            self._pyboy,
            action_frames=self.action_frames,
            render=self.window != "null",
            actions=self.actions,
        )

    def _check_paths(self) -> None:
        if not self.rom_path.exists():
            raise FileNotFoundError(f"ROM nao encontrada: {self.rom_path}")
        if not self.state_path.exists():
            raise FileNotFoundError(
                f"State da fase nao encontrado: {self.state_path}. "
                "Use scripts/manual_control.py para criar esse state primeiro."
            )
        if self.symbols_path and not self.symbols_path.exists():
            raise FileNotFoundError(f"Arquivo de simbolos nao encontrado: {self.symbols_path}")

    def _snapshot(self) -> GameSnapshot:
        if self._reader is None:
            raise RuntimeError("Ambiente ainda nao foi inicializado. Chame reset() primeiro.")
        return self._reader.snapshot()

    def _build_observation_space(self) -> spaces.Space:
        if self.observation_mode == "coords":
            return spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)

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
                    "position": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
                    "map": spaces.Box(
                        low=0,
                        high=255,
                        shape=(1, self.coords_pad * 4, self.coords_pad * 4),
                        dtype=np.uint8,
                    ),
                    "recent_actions": spaces.MultiDiscrete([len(self.actions)] * self.frame_stacks),
                }
            )

        raise ValueError("observation_mode deve ser 'coords', 'screen' ou 'multi'")

    def _observation(self, snapshot: GameSnapshot) -> Observation:
        if self.observation_mode == "coords":
            return np.array([snapshot.map_id, snapshot.x, snapshot.y], dtype=np.uint8)

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
            "position": self._position_features(snapshot),
            "map": self._local_explore_map(snapshot)[np.newaxis, :, :],
            "recent_actions": self._recent_actions.copy(),
        }

    def _screen_observation(self) -> np.ndarray:
        screen = np.asarray(self._pyboy.screen.ndarray[:, :, :3])
        grayscale = screen.mean(axis=2).astype(np.uint8)
        return grayscale[::2, ::2][np.newaxis, :, :].copy()

    def _record_position(self, snapshot: GameSnapshot) -> None:
        if snapshot.in_battle:
            return
        self._seen_coords[snapshot.position] = self._seen_coords.get(snapshot.position, 0) + 1

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

    def _position_features(self, snapshot: GameSnapshot) -> np.ndarray:
        target_map = self.phase.target_position_map
        if target_map is None:
            target_map = self.phase.target_map

        target_x = snapshot.x if self.phase.target_x is None else self.phase.target_x
        target_y = snapshot.y if self.phase.target_y is None else self.phase.target_y
        success_map = snapshot.map_id if self.phase.target_map is None else self.phase.target_map

        same_position_map = 1.0 if target_map is not None and snapshot.map_id == target_map else 0.0
        same_success_map = 1.0 if snapshot.map_id == success_map else 0.0
        dx = np.clip((target_x - snapshot.x) / 20.0, -1.0, 1.0)
        dy = np.clip((target_y - snapshot.y) / 20.0, -1.0, 1.0)

        return np.array(
            [
                np.clip(snapshot.map_id / 255.0, 0.0, 1.0),
                np.clip(snapshot.x / 255.0, 0.0, 1.0),
                np.clip(snapshot.y / 255.0, 0.0, 1.0),
                np.clip(success_map / 255.0, 0.0, 1.0),
                dx,
                dy,
                same_position_map,
                same_success_map,
            ],
            dtype=np.float32,
        )

    def _update_recent_screens(self, screen: np.ndarray) -> None:
        self._recent_screens = np.roll(self._recent_screens, 1, axis=0)
        self._recent_screens[0] = screen

    def _update_recent_actions(self, action: int) -> None:
        self._recent_actions = np.roll(self._recent_actions, 1)
        self._recent_actions[0] = int(action)

    def _is_blocked_move(
        self,
        action: int,
        previous: GameSnapshot,
        current: GameSnapshot,
    ) -> bool:
        if action_name(action, self.actions) not in {"up", "down", "left", "right"}:
            return False
        if previous.in_battle or current.in_battle:
            return False
        return previous.position == current.position

    def _fourier_encode(self, value: float) -> np.ndarray:
        return np.sin(value * 2 ** np.arange(self.enc_freqs)).astype(np.float32)

    def _info(self, snapshot: GameSnapshot, reward: float, success: bool) -> dict[str, Any]:
        return {
            "phase": self.phase.id,
            "phase_name": self.phase.name,
            "step_count": self._step_count,
            "success": success,
            "reward": reward,
            "seen_coords": len(self._seen_coords),
            "snapshot": snapshot.to_info(),
        }
