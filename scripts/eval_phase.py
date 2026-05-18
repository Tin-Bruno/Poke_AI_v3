from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs import PokemonRedEnv
from phases import get_phase
from project_config import env_float, env_int, env_str, load_dotenv

warnings.filterwarnings("ignore", message="Using SDL2 binaries from pysdl2-dll.*")


def should_stop_eval() -> bool:
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


def interruptible_delay(seconds: float) -> bool:
    if seconds <= 0:
        return False

    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        if should_stop_eval():
            return True
        time.sleep(min(0.05, max(0.0, end_time - time.monotonic())))
    return False


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
    parser.add_argument(
        "--save-wait-frames",
        type=int,
        default=env_int("POKE_SAVE_WAIT_FRAMES", 90),
        help="Frames para estabilizar o jogo antes de salvar o proximo state.",
    )
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

    if args.observation_mode != "screen":
        eval_raw_env(args, phase, model_path)
        return

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
        model = load_model(model_path, phase)
        ensure_model_compatible(model, env, phase)

        obs = env.reset()
        final_info = {}
        stop_reason = "limite de steps"

        for _ in range(args.steps):
            if should_stop_eval():
                stop_reason = "interrompido com q"
                break

            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, rewards, dones, infos = env.step(action)
            final_info = infos[0]
            if final_info.get("success"):
                stop_reason = "sucesso"
                if args.save_success_state:
                    raise SystemExit(
                        "Salvar state de sucesso em observation_mode=screen nao e suportado "
                        "neste fluxo. Use --observation-mode coords para salvar o proximo state."
                    )
                break
            if dones[0]:
                stop_reason = "episodio encerrado"
                break
            if interruptible_delay(args.delay):
                stop_reason = "interrompido com q"
                break

        print_final_summary(phase, final_info, stop_reason, args.full_info)
    finally:
        env.close()


def eval_raw_env(args: argparse.Namespace, phase, model_path: str | Path) -> None:
    env = PokemonRedEnv(
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
    try:
        model = load_model(model_path, phase)
        ensure_model_compatible(model, env, phase)

        obs, _ = env.reset()
        final_info = {}
        stop_reason = "limite de steps"
        saved_state = None

        for _ in range(args.steps):
            if should_stop_eval():
                stop_reason = "interrompido com q"
                break

            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _reward, terminated, truncated, info = env.step(action)
            final_info = info

            if final_info.get("success"):
                stop_reason = "sucesso"
                if args.save_success_state:
                    env.wait(args.save_wait_frames)
                    env.save_state(Path(args.save_success_state))
                    saved_state = args.save_success_state
                break
            if terminated or truncated:
                stop_reason = "timeout sem progresso" if info.get("no_progress_timeout") else "episodio encerrado"
                break
            if interruptible_delay(args.delay):
                stop_reason = "interrompido com q"
                break

        print_final_summary(phase, final_info, stop_reason, args.full_info, saved_state)
    finally:
        env.close()


def load_model(model_path: str | Path, phase) -> PPO:
    try:
        return PPO.load(model_path)
    except ValueError as exc:
        raise SystemExit(
            "Nao foi possivel carregar o modelo. "
            "Isso normalmente acontece depois de mudar observacao, acoes ou rewards. "
            f"Treine novamente com: python scripts/train_phase.py --phase {phase.id} --timesteps 100000\n"
            f"Detalhe: {exc}"
        ) from exc


def ensure_model_compatible(model: PPO, env: PokemonRedEnv, phase) -> None:
    if model.action_space != env.action_space:
        raise SystemExit(
            "Modelo incompatível com a configuração atual da fase. "
            "Isso normalmente acontece depois de mudar o conjunto de ações. "
            f"Treine novamente com: python scripts/train_phase.py --phase {phase.id} "
            "--timesteps 100000 --observation-mode coords --window null"
        )
    if model.observation_space != env.observation_space:
        raise SystemExit(
            "Modelo incompatível com a observação atual da fase. "
            "Isso normalmente acontece depois de mudar --observation-mode. "
            f"Treine novamente com: python scripts/train_phase.py --phase {phase.id} "
            "--timesteps 100000 --observation-mode coords --window null"
        )


def print_final_summary(
    phase,
    info: dict,
    stop_reason: str,
    full_info: bool,
    saved_state: str | None = None,
) -> None:
    compact = compact_info(info)
    status = "SUCESSO" if compact["success"] else "PAROU"
    reward = compact["reward_scaled"]
    reward_text = "n/a" if reward is None else f"{reward:.3f}"

    print("")
    print("=== Avaliacao final ===")
    print(f"Fase:        {phase.id} - {phase.name}")
    print(f"Status:      {status}")
    print(f"Motivo:      {stop_reason}")
    print(f"Passos:      {compact['step_count']}")
    print(f"Posicao:     map={compact['map_id']} x={compact['x']} y={compact['y']}")
    print(f"Acao final:  {compact['action']}")
    print(f"Reward:      {reward_text}")
    print(f"Vistos:      {compact['seen_coords']} coords")
    if compact["no_progress_timeout"]:
        print("Timeout:     sem progresso")
    if saved_state:
        print(f"State salvo: {saved_state}")
    print("=======================")

    if full_info:
        print(f"Info completo: {info}")


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
