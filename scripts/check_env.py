from __future__ import annotations

import argparse

from stable_baselines3.common.env_checker import check_env

from poke_ai_v3 import PokemonRedEnv
from poke_ai_v3.config import env_int, env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Valida o ambiente Gymnasium do Pokemon.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"), help="Caminho para a ROM .gb/.gbc.")
    parser.add_argument(
        "--state",
        default=env_str("POKE_STATE_PATH"),
        help="Estado PyBoy opcional para reset inicial.",
    )
    parser.add_argument(
        "--symbols",
        default=env_str("POKE_SYMBOLS_PATH"),
        help="Arquivo .sym opcional do pokered.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Acoes aleatorias apos validar.")
    parser.add_argument(
        "--observation-mode",
        choices=("screen", "multi"),
        default=env_str("POKE_OBSERVATION_MODE", "multi"),
        help="screen usa pixels; multi inclui memoria e mapa local.",
    )
    parser.add_argument(
        "--frame-stacks",
        type=int,
        default=env_int("POKE_FRAME_STACKS", 3),
        help="Frames recentes no modo multi.",
    )
    parser.add_argument(
        "--window",
        default=env_str("POKE_WINDOW", "null"),
        help='Janela PyBoy: "null" ou "SDL2".',
    )
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou defina POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    env = PokemonRedEnv(
        rom_path=args.rom,
        state_path=args.state,
        symbols_path=args.symbols,
        window=args.window,
        observation_mode=args.observation_mode,
        frame_stacks=args.frame_stacks,
    )
    try:
        check_env(env, warn=True)
        obs, info = env.reset()
        total_reward = 0.0
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            if terminated or truncated:
                obs, info = env.reset()

        print("Ambiente OK")
        print(f"Ultima observacao: shape={obs.shape}, dtype={obs.dtype}")
        print(f"Recompensa acumulada no smoke test: {total_reward:.3f}")
        print(f"Ultimo snapshot: {info.get('snapshot')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
