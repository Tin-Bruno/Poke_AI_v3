from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import PokemonRedEnv
from envs.freeplay_env import FreeplayConfig, FreeplayPokemonRedEnv
from phases import get_phase
from project_config import env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Mede steps/s do ambiente.")
    parser.add_argument("--mode", choices=("phase", "freeplay"), default="phase")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--state", default="states/freeplay_start.state")
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--window", choices=("null", "SDL2"), default="null")
    parser.add_argument("--observation-mode", choices=("coords", "ram", "multi"), default="coords")
    parser.add_argument("--action-set", choices=("simple", "combo"), default="simple")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def make_env(args: argparse.Namespace):
    if args.mode == "freeplay":
        return FreeplayPokemonRedEnv(
            FreeplayConfig(
                rom_path=args.rom,
                state_path=args.state,
                symbols_path=args.symbols,
                window=args.window,
                action_set=args.action_set,
                observation_mode=args.observation_mode,
            )
        )

    return PokemonRedEnv(
        rom_path=args.rom,
        phase=get_phase(args.phase),
        states_dir=args.states_dir,
        symbols_path=args.symbols,
        window=args.window,
        observation_mode=args.observation_mode,
    )


def main() -> None:
    args = parse_args()
    env = make_env(args)
    try:
        obs, _info = env.reset()
        start = time.perf_counter()
        completed = 0
        for completed in range(1, args.steps + 1):
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                obs, _info = env.reset()
        elapsed = max(time.perf_counter() - start, 1e-9)
        print("")
        print("=== Benchmark ===")
        print(f"Modo:       {args.mode}")
        print(f"Steps:      {completed}")
        print(f"Tempo:      {elapsed:.2f}s")
        print(f"Steps/s:    {completed / elapsed:.1f}")
        print("=================")
    finally:
        env.close()


if __name__ == "__main__":
    main()
