from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.env_checker import check_env

from envs import PokemonRedEnv
from project_config import env_float, env_int, env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Smoke test de uma fase.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default=env_str("POKE_WINDOW", "null"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--observation-mode",
        choices=("coords", "ram", "screen", "multi"),
        default=env_str("POKE_OBSERVATION_MODE", "coords"),
    )
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_ACTION_FRAMES", 12))
    parser.add_argument("--reward-scale", type=float, default=env_float("POKE_REWARD_SCALE", 1.0))
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    env = PokemonRedEnv(
        rom_path=args.rom,
        phase=args.phase,
        states_dir=args.states_dir,
        symbols_path=args.symbols,
        window=args.window,
        action_frames=args.action_frames,
        observation_mode=args.observation_mode,
        reward_scale=args.reward_scale,
    )
    try:
        check_env(env, warn=True)
        obs, info = env.reset()
        total = 0.0
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            if terminated or truncated:
                break
        print("Ambiente OK")
        print(f"Fase: {info['phase']} - {info['phase_name']}")
        print(f"Reward acumulado: {total:.3f}")
        print(f"Info final: {info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
