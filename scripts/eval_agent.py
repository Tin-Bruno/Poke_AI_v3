from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from poke_ai_v3 import PokemonRedEnv
from poke_ai_v3.config import env_int, env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Roda um modelo treinado em Pokemon Red/Blue.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"), help="Caminho para a ROM .gb/.gbc.")
    parser.add_argument("--model", required=True, help="Arquivo .zip salvo pelo Stable-Baselines3.")
    parser.add_argument(
        "--observation-mode",
        choices=("screen", "multi"),
        default=env_str("POKE_OBSERVATION_MODE", "multi"),
        help="Use o mesmo modo usado no treino.",
    )
    parser.add_argument(
        "--frame-stacks",
        type=int,
        default=env_int("POKE_FRAME_STACKS", 3),
        help="Frames recentes no modo multi.",
    )
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
    parser.add_argument(
        "--window",
        default=env_str("POKE_WINDOW", "SDL2"),
        help='Janela PyBoy: "SDL2" ou "null".',
    )
    parser.add_argument("--steps", type=int, default=10_000, help="Passos de avaliacao.")
    parser.add_argument(
        "--action-frames",
        type=int,
        default=env_int("POKE_ACTION_FRAMES", 12),
        help="Frames por acao.",
    )
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou defina POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()

    def make_env() -> PokemonRedEnv:
        return PokemonRedEnv(
            rom_path=args.rom,
            state_path=args.state,
            symbols_path=args.symbols,
            window=args.window,
            action_frames=args.action_frames,
            observation_mode=args.observation_mode,
            frame_stacks=args.frame_stacks,
        )

    env = DummyVecEnv([make_env])
    if args.observation_mode == "screen":
        env = VecFrameStack(env, n_stack=4, channels_order="first")
    model = PPO.load(args.model, env=env)

    obs = env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones[0]:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
