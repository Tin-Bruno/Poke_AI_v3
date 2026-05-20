from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs import PokemonRedEnv
from phases.phase_config import PHASES, PhaseConfig, get_phase
from project_config import env_float, env_int, env_str, load_dotenv
from scripts.eval_phase import action_index, compact_info, ensure_model_compatible, run_save_actions


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Avalia varias fases em sequencia.")
    parser.add_argument("phases", nargs="*", help="Ex: phase1 phase2 phase3")
    parser.add_argument("--through", default=None, help="Roda da primeira fase ate esta fase. Ex: --through phase5")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--sequence-dir", default="logs/eval_sequence_states")
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default=env_str("POKE_WINDOW", "null"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--observation-mode",
        choices=("auto", "coords", "ram", "screen", "multi"),
        default="auto",
        help="Use auto para inferir o modo pela observation_space de cada modelo.",
    )
    parser.add_argument("--frame-stacks", type=int, default=env_int("POKE_FRAME_STACKS", 3))
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_ACTION_FRAMES", 12))
    parser.add_argument("--reward-scale", type=float, default=env_float("POKE_REWARD_SCALE", 1.0))
    parser.add_argument("--save-wait-frames", type=int, default=env_int("POKE_SAVE_WAIT_FRAMES", 90))
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    if args.through and args.phases:
        parser.error("use --through ou a lista de fases, nao os dois ao mesmo tempo")
    if not args.through and not args.phases:
        parser.error("informe as fases ou use --through phaseX")
    return args


def phase_ids_from_args(args: argparse.Namespace) -> list[str]:
    if args.through:
        ids = list(PHASES)
        if args.through not in PHASES:
            raise SystemExit(f"Fase desconhecida: {args.through}. Fases validas: {', '.join(ids)}")
        return ids[: ids.index(args.through) + 1]
    return args.phases


def main() -> None:
    args = parse_args()
    phase_ids = phase_ids_from_args(args)
    phases = [get_phase(phase_id) for phase_id in phase_ids]

    sequence_dir = Path(args.sequence_dir)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    first_state = Path(args.states_dir) / phases[0].state
    current_state = sequence_dir / phases[0].state
    shutil.copy2(first_state, current_state)

    print("=== Eval sequence ===")
    print(f"Fases: {' -> '.join(phase.id for phase in phases)}")
    print(f"States temporarios: {sequence_dir}")
    print("")

    final_info: dict = {}
    for index, phase in enumerate(phases):
        next_phase = phases[index + 1] if index + 1 < len(phases) else None
        output_state = sequence_dir / (next_phase.state if next_phase else f"{phase.id}_success.state")
        success, final_info = run_phase(args, phase, current_state, output_state)
        if not success:
            print("")
            print("Sequencia interrompida.")
            raise SystemExit(1)
        current_state = output_state

    compact = compact_info(final_info)
    print("")
    print("=== Sequencia OK ===")
    print(f"Fase final: {compact['phase']}")
    print(f"Posicao final: map={compact['map_id']} x={compact['x']} y={compact['y']}")
    print(f"State final: {current_state}")


def run_phase(args: argparse.Namespace, phase: PhaseConfig, state_path: Path, output_state: Path) -> tuple[bool, dict]:
    model_path = Path(phase.model)
    if not phase.scripted_actions and not model_path.exists():
        raise SystemExit(f"Modelo nao encontrado para {phase.id}: {model_path}")

    model = None if phase.scripted_actions else PPO.load(model_path)
    if args.observation_mode == "auto":
        observation_mode = "coords" if model is None else infer_observation_mode(model)
    else:
        observation_mode = args.observation_mode
    env = PokemonRedEnv(
        rom_path=args.rom,
        phase=phase,
        state_path=state_path,
        symbols_path=args.symbols,
        window=args.window,
        action_frames=args.action_frames,
        observation_mode=observation_mode,
        frame_stacks=args.frame_stacks,
        reward_scale=args.reward_scale,
    )
    try:
        if model is not None:
            ensure_model_compatible(model, env, phase)

        obs, _ = env.reset()
        final_info = {}
        stop_reason = "limite de steps"
        scripted_actions = list(phase.scripted_actions)

        for _ in range(args.steps):
            if scripted_actions:
                action = action_index(env, phase, scripted_actions.pop(0))
            elif phase.scripted_actions:
                stop_reason = "script concluido"
                break
            else:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _reward, terminated, truncated, info = env.step(action)
            final_info = info
            if info.get("success"):
                run_save_actions(env, phase)
                env.wait(args.save_wait_frames)
                env.save_state(output_state)
                stop_reason = "sucesso"
                break
            if terminated or truncated:
                stop_reason = "timeout sem progresso" if info.get("no_progress_timeout") else "episodio encerrado"
                break

        print_phase_summary(phase, final_info, stop_reason, output_state, observation_mode)
        return bool(final_info.get("success")), final_info
    finally:
        env.close()


def infer_observation_mode(model: PPO) -> str:
    space = model.observation_space
    if hasattr(space, "spaces"):
        return "multi"

    shape = getattr(space, "shape", None)
    dtype = str(getattr(space, "dtype", ""))
    if shape == (3,) and dtype == "uint8":
        return "coords"
    if shape == (6,) and dtype == "float32":
        return "ram"
    if shape and len(shape) == 3:
        return "screen"

    raise SystemExit(f"Nao consegui inferir observation-mode para observation_space={space}")


def print_phase_summary(
    phase: PhaseConfig,
    info: dict,
    stop_reason: str,
    output_state: Path,
    observation_mode: str,
) -> None:
    compact = compact_info(info)
    status = "OK" if compact["success"] else "FALHOU"
    print(
        f"{phase.id}: {status} | {stop_reason} | "
        f"map={compact['map_id']} x={compact['x']} y={compact['y']} | "
        f"action={compact['action']} | steps={compact['step_count']} | obs={observation_mode}"
    )
    if compact["success"]:
        print(f"  state: {output_state}")


if __name__ == "__main__":
    main()
