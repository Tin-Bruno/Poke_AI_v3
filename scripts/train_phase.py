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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from envs import PokemonRedEnv
from phases import get_phase
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

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True
        if not should_stop_training():
            return True
        print("")
        print("Treino interrompido pelo usuario com Q. Salvando modelo...")
        return False


class PhaseStatsCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in (
                "success",
                "seen_coords",
                "reward_scaled",
                "steps_since_progress",
                "waypoint_index",
                "dialog",
                "party",
            ):
                if key in info:
                    self.logger.record(f"phase/{key}", info[key])
            snapshot = info.get("snapshot", {})
            for key in ("map_id", "x", "y", "event_count", "party_count", "hp_fraction"):
                if key in snapshot:
                    self.logger.record(f"ram/{key}", snapshot[key])
        return True


def parse_args() -> argparse.Namespace:
    load_dotenv()
    runs_dir = Path(env_str("POKE_RUNS_DIR", "runs"))
    parser = argparse.ArgumentParser(description="Treina uma fase especifica.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default=env_str("POKE_WINDOW", "null"))
    parser.add_argument("--timesteps", type=int, default=env_int("POKE_TIMESTEPS", 100_000))
    parser.add_argument("--n-envs", type=int, default=env_int("POKE_N_ENVS", 1))
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default=env_str("POKE_VEC_ENV", "dummy"))
    parser.add_argument(
        "--observation-mode",
        choices=("coords", "ram", "screen", "multi"),
        default=env_str("POKE_OBSERVATION_MODE", "coords"),
    )
    parser.add_argument("--frame-stacks", type=int, default=env_int("POKE_FRAME_STACKS", 3))
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_ACTION_FRAMES", 12))
    parser.add_argument("--warmup-frames", type=int, default=env_int("POKE_WARMUP_FRAMES", 0))
    parser.add_argument("--reward-scale", type=float, default=env_float("POKE_REWARD_SCALE", 1.0))
    parser.add_argument("--max-no-progress-steps", type=int, default=env_int("POKE_MAX_NO_PROGRESS_STEPS", 200))
    parser.add_argument("--n-steps", type=int, default=env_int("POKE_PPO_N_STEPS", 512))
    parser.add_argument("--batch-size", type=int, default=env_int("POKE_PPO_BATCH_SIZE", 64))
    parser.add_argument("--learning-rate", type=float, default=env_float("POKE_PPO_LEARNING_RATE", 3e-4))
    parser.add_argument("--gamma", type=float, default=env_float("POKE_PPO_GAMMA", 0.95))
    parser.add_argument("--n-epochs", type=int, default=env_int("POKE_PPO_N_EPOCHS", 10))
    parser.add_argument("--ent-coef", type=float, default=env_float("POKE_PPO_ENT_COEF", 0.0))
    parser.add_argument(
        "--stop-key-check-freq",
        type=int,
        default=env_int("POKE_STOP_KEY_CHECK_FREQ", 500),
        help="Frequencia em timesteps para checar se Q foi apertado no terminal.",
    )
    parser.add_argument("--checkpoint-dir", default=runs_dir / "checkpoints")
    parser.add_argument("--tensorboard-dir", default=runs_dir / "tensorboard")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    checkpoint_dir = Path(args.checkpoint_dir) / phase.id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(phase.model).parent.mkdir(parents=True, exist_ok=True)

    def make_env() -> Monitor:
        env = PokemonRedEnv(
            rom_path=args.rom,
            phase=phase,
            states_dir=args.states_dir,
            symbols_path=args.symbols,
            window=args.window,
            action_frames=args.action_frames,
            warmup_frames=args.warmup_frames,
            observation_mode=args.observation_mode,
            frame_stacks=args.frame_stacks,
            reward_scale=args.reward_scale,
            max_no_progress_steps=args.max_no_progress_steps,
        )
        return Monitor(env)

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and args.n_envs > 1 else DummyVecEnv
    env = vec_cls([make_env for _ in range(args.n_envs)])
    if args.observation_mode == "screen":
        env = VecFrameStack(env, n_stack=4, channels_order="first")

    if args.observation_mode == "multi":
        policy = "MultiInputPolicy"
    elif args.observation_mode == "screen":
        policy = "CnnPolicy"
    else:
        policy = "MlpPolicy"
    batch_size = min(args.batch_size, max(args.n_steps * args.n_envs, 2))

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(10_000 // max(1, args.n_envs), 1),
                save_path=str(checkpoint_dir),
                name_prefix=phase.id,
            ),
            PhaseStatsCallback(),
            StopOnQCallback(check_freq=args.stop_key_check_freq),
        ]
    )

    model = PPO(
        policy,
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        tensorboard_log=args.tensorboard_dir,
    )
    try:
        print("Pressione Q no terminal para interromper e salvar.")
        model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name=phase.id)
    finally:
        model.save(phase.model)
        env.close()
        print(f"Modelo salvo: {phase.model}")


if __name__ == "__main__":
    main()
