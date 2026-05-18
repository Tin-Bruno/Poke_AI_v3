from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs import PokemonRedEnv
from phases import get_phase
from project_config import env_float, env_int, env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Avalia uma fase treinada.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default=env_str("POKE_WINDOW", "SDL2"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--observation-mode",
        choices=("coords", "screen", "multi"),
        default=env_str("POKE_OBSERVATION_MODE", "coords"),
    )
    parser.add_argument("--frame-stacks", type=int, default=env_int("POKE_FRAME_STACKS", 3))
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_ACTION_FRAMES", 12))
    parser.add_argument("--reward-scale", type=float, default=env_float("POKE_REWARD_SCALE", 1.0))
    parser.add_argument("--delay", type=float, default=0.0, help="Pausa em segundos entre acoes para assistir na janela.")
    parser.add_argument("--save-success-state", default=None)
    parser.add_argument("--stochastic", action="store_true", help="Usa politica estocastica em vez da deterministica.")
    parser.add_argument("--full-info", action="store_true", help="Imprime o info completo, incluindo arrays.")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    model_path = args.model or phase.model

    def make_env() -> PokemonRedEnv:
        return PokemonRedEnv(
            rom_path=args.rom,
            phase=phase,
            states_dir=args.states_dir,
            symbols_path=args.symbols,
            window=args.window,
            action_frames=args.action_frames,
            observation_mode=args.observation_mode,
            frame_stacks=args.frame_stacks,
            reward_scale=args.reward_scale,
        )

    env = DummyVecEnv([make_env])
    if args.observation_mode == "screen":
        env = VecFrameStack(env, n_stack=4, channels_order="first")

    try:
        model = PPO.load(model_path, env=env)
    except ValueError as exc:
        env.close()
        raise SystemExit(
            "Nao foi possivel carregar o modelo com a observacao atual. "
            "Isso normalmente acontece depois de mudar observacao, acoes ou rewards. "
            f"Treine novamente com: python scripts/train_phase.py --phase {phase.id} --timesteps 100000\n"
            f"Detalhe: {exc}"
        ) from exc
    obs = env.reset()
    final_info = {}

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, rewards, dones, infos = env.step(action)
        final_info = infos[0]
        if final_info.get("success"):
            print(f"Sucesso na fase {phase.id}: {phase.name}")
            if args.save_success_state:
                raw_env = unwrap_raw_env(env)
                raw_env.save_state(Path(args.save_success_state))
                print(f"Proximo state salvo: {args.save_success_state}")
            break
        if dones[0]:
            break
        if args.delay > 0:
            time.sleep(args.delay)

    info_to_print = final_info if args.full_info else compact_info(final_info)
    print(f"Info final: {info_to_print}")
    env.close()


def unwrap_raw_env(env):
    if hasattr(env, "venv"):
        return env.venv.envs[0]
    return env.envs[0]


def compact_info(info: dict) -> dict:
    snapshot = info.get("snapshot", {})
    return {
        "phase": info.get("phase"),
        "step_count": info.get("step_count"),
        "success": info.get("success"),
        "reward": info.get("reward"),
        "reward_scaled": info.get("reward_scaled"),
        "seen_coords": info.get("seen_coords"),
        "steps_since_progress": info.get("steps_since_progress"),
        "action": info.get("action"),
        "blocked_move": info.get("blocked_move"),
        "no_progress_timeout": info.get("no_progress_timeout"),
        "map_id": snapshot.get("map_id"),
        "x": snapshot.get("x"),
        "y": snapshot.get("y"),
        "event_count": snapshot.get("event_count"),
    }


if __name__ == "__main__":
    main()
