from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.freeplay_env import FreeplayConfig, FreeplayPokemonRedEnv
from project_config import env_float, env_int, env_str, load_dotenv


def should_stop_training() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False

    if not msvcrt.kbhit():
        return False

    key = msvcrt.getch()
    if key in (b"\x00", b"\xe0"):
        if msvcrt.kbhit():
            msvcrt.getch()
        return False

    return key.decode("utf-8", errors="ignore").lower() == "q"


class StopOnQCallback(BaseCallback):
    def __init__(self, check_freq: int = 500) -> None:
        super().__init__()
        self.check_freq = max(1, check_freq)
        self.interrupted = False

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True
        if not should_stop_training():
            return True
        self.interrupted = True
        print("")
        print("Freeplay interrompido com Q. Salvando modelo interrompido...")
        return False


class FreeplayStatsCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in (
                "reward_location",
                "reward_map",
                "reward_events",
                "reward_badges",
                "reward_party",
                "reward_survival",
                "reward_repeat_penalty",
                "reward_step_penalty",
                "seen_locations",
                "seen_maps",
                "steps_since_progress",
            ):
                if key in info:
                    self.logger.record(f"freeplay/{key}", info[key])
            for key in ("map_id", "x", "y", "event_count", "party_count", "hp_fraction"):
                if key in info:
                    self.logger.record(f"ram/{key}", info[key])
        return True


def interrupted_model_path(save_path: str | Path) -> Path:
    path = Path(save_path)
    suffix = path.suffix or ".zip"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_interrompido{suffix}")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    runs_dir = Path(env_str("POKE_RUNS_DIR", "runs"))
    parser = argparse.ArgumentParser(description="Treina modo freeplay sem curriculo por fases.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--state", default="states/phase9_start.state")
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", choices=("null", "SDL2"), default="null")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy")
    parser.add_argument("--start-method", choices=("spawn", "forkserver", "fork"), default="spawn")
    parser.add_argument("--action-frames", type=int, default=24)
    parser.add_argument("--release-frames", type=int, default=6)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--stagnation-steps", type=int, default=4_000)
    parser.add_argument("--emulation-speed", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=env_float("POKE_FREEPLAY_LR", 2.5e-4))
    parser.add_argument("--gamma", type=float, default=env_float("POKE_FREEPLAY_GAMMA", 0.997))
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-path", default="models/pokemon_red_freeplay.zip")
    parser.add_argument("--checkpoint-dir", default=runs_dir / "freeplay_checkpoints")
    parser.add_argument("--tensorboard-dir", default=runs_dir / "tensorboard")
    parser.add_argument("--stop-key-check-freq", type=int, default=env_int("POKE_STOP_KEY_CHECK_FREQ", 500))
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    if args.n_envs > 1 and args.window != "null":
        print("Janela SDL2 desligada no treino com varios ambientes. Use --n-envs 1 para assistir.")
        args.window = "null"
    return args


def main() -> None:
    args = parse_args()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def make_env() -> Monitor:
        config = FreeplayConfig(
            rom_path=args.rom,
            state_path=args.state,
            symbols_path=args.symbols,
            window=args.window,
            action_frames=args.action_frames,
            release_frames=args.release_frames,
            warmup_frames=args.warmup_frames,
            max_steps=args.max_steps,
            stagnation_steps=args.stagnation_steps,
            emulation_speed=args.emulation_speed,
        )
        return Monitor(FreeplayPokemonRedEnv(config))

    if args.vec_env == "subproc" and args.n_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(args.n_envs)], start_method=args.start_method)
    else:
        env = DummyVecEnv([make_env for _ in range(args.n_envs)])

    stop_callback = StopOnQCallback(check_freq=args.stop_key_check_freq)
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(10_000 // max(1, args.n_envs), 1),
                save_path=str(checkpoint_dir),
                name_prefix="pokemon_red_freeplay",
            ),
            FreeplayStatsCallback(),
            stop_callback,
        ]
    )

    batch_size = min(args.batch_size, max(args.n_steps * args.n_envs, 2))
    if args.resume:
        model = PPO.load(args.resume, env=env, device="auto")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            tensorboard_log=args.tensorboard_dir,
            device="auto",
        )

    interrupted = False
    try:
        print("Pressione Q no terminal para interromper e salvar como modelo interrompido.")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="freeplay",
            reset_num_timesteps=args.resume is None,
        )
        interrupted = stop_callback.interrupted
    except KeyboardInterrupt:
        interrupted = True
        print("")
        print("Freeplay interrompido pelo terminal. Salvando modelo interrompido...")
    finally:
        save_path = interrupted_model_path(args.save_path) if interrupted else Path(args.save_path)
        model.save(str(save_path))
        env.close()
        print(f"Modelo salvo: {save_path}")


if __name__ == "__main__":
    main()
