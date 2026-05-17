from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from poke_ai_v3 import PokemonRedEnv
from poke_ai_v3.callbacks import PokeStatsCallback
from poke_ai_v3.config import env_int, env_str, load_dotenv
from poke_ai_v3.rewards import RewardConfig


def parse_args() -> argparse.Namespace:
    load_dotenv()
    runs_dir = Path(env_str("POKE_RUNS_DIR", "runs"))
    parser = argparse.ArgumentParser(description="Treina PPO para Pokemon Red/Blue.")
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
    parser.add_argument(
        "--window",
        default=env_str("POKE_WINDOW", "null"),
        help='Janela PyBoy: "null" ou "SDL2".',
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=env_int("POKE_TIMESTEPS", 100_000),
        help="Total de passos de treino.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=env_int("POKE_N_ENVS", 1),
        help="Numero de ambientes em paralelo.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default=env_str("POKE_VEC_ENV", "dummy"),
        help="Tipo de VecEnv. Use subproc para varios processos.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("screen", "multi"),
        default=env_str("POKE_OBSERVATION_MODE", "multi"),
        help="screen usa CnnPolicy; multi usa MultiInputPolicy estilo V2.",
    )
    parser.add_argument(
        "--frame-stacks",
        type=int,
        default=env_int("POKE_FRAME_STACKS", 3),
        help="Frames recentes no modo multi.",
    )
    parser.add_argument(
        "--action-frames",
        type=int,
        default=env_int("POKE_ACTION_FRAMES", 12),
        help="Frames por acao.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=env_int("POKE_WARMUP_FRAMES", 120),
        help="Frames apos reset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=env_int("POKE_MAX_STEPS", 2048 * 80),
        help="Limite de passos por episodio.",
    )
    parser.add_argument(
        "--max-no-progress-steps",
        type=int,
        default=env_int("POKE_MAX_NO_PROGRESS_STEPS", 2000),
        help="Trunca episodio apos tantos passos sem progresso.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=float(env_str("POKE_REWARD_SCALE", "0.5")),
        help="Multiplicador final de recompensa.",
    )
    parser.add_argument(
        "--explore-weight",
        type=float,
        default=float(env_str("POKE_EXPLORE_WEIGHT", "0.25")),
        help="Peso da recompensa de exploracao.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=env_int("POKE_PPO_N_STEPS", 2048),
        help="Passos por rollout PPO em cada ambiente.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=env_int("POKE_PPO_BATCH_SIZE", 512),
        help="Batch size do PPO.",
    )
    parser.add_argument("--checkpoint-dir", default=runs_dir / "checkpoints", help="Pasta de checkpoints.")
    parser.add_argument("--model-dir", default=runs_dir / "models", help="Pasta do modelo final.")
    parser.add_argument("--tensorboard-dir", default=runs_dir / "tensorboard", help="Logs TensorBoard.")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou defina POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    model_dir = Path(args.model_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    reward_config = RewardConfig(
        max_no_progress_steps=args.max_no_progress_steps,
        reward_scale=args.reward_scale,
        explore_weight=args.explore_weight,
    )

    def make_env() -> PokemonRedEnv:
        return PokemonRedEnv(
            rom_path=args.rom,
            state_path=args.state,
            symbols_path=args.symbols,
            window=args.window,
            action_frames=args.action_frames,
            warmup_frames=args.warmup_frames,
            max_steps=args.max_steps,
            observation_mode=args.observation_mode,
            frame_stacks=args.frame_stacks,
            reward_config=reward_config,
        )

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and args.n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env for _ in range(args.n_envs)])
    if args.observation_mode == "screen":
        env = VecFrameStack(env, n_stack=4, channels_order="first")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // max(1, args.n_envs), 1),
        save_path=str(checkpoint_dir),
        name_prefix="poke_red_ppo",
    )
    callbacks = CallbackList([checkpoint_callback, PokeStatsCallback()])

    policy = "MultiInputPolicy" if args.observation_mode == "multi" else "CnnPolicy"
    rollout_size = max(args.n_steps * args.n_envs, 2)
    batch_size = min(args.batch_size, rollout_size)

    model = PPO(
        policy,
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=1,
        gamma=0.997,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        tensorboard_log=args.tensorboard_dir,
    )
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(model_dir / "poke_red_ppo")
    env.close()


if __name__ == "__main__":
    main()
