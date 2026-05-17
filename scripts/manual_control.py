from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.step_handler import ACTIONS, StepHandler
from memory import PokemonRedRamReader
from phases import get_phase
from project_config import env_int, env_str, load_dotenv


KEY_TO_ACTION = {
    "w": "up",
    "s": "down",
    "a": "left",
    "d": "right",
    "j": "a",
    "k": "b",
    "u": "start",
    "i": "select",
    ".": "noop",
}


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Controle manual para criar states e inspecionar RAM.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"), help="Caminho da ROM.")
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"), help="Fase base.")
    parser.add_argument("--state", default=None, help="State inicial opcional.")
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default=env_str("POKE_WINDOW", "SDL2"))
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_ACTION_FRAMES", 12))
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    state_path = Path(args.state) if args.state else Path(args.states_dir) / phase.state

    from pyboy import PyBoy

    kwargs = {"window": args.window, "sound_emulated": False}
    if args.symbols:
        kwargs["symbols"] = args.symbols

    pyboy = PyBoy(args.rom, **kwargs)
    pyboy.set_emulation_speed(0)
    if state_path.exists():
        with state_path.open("rb") as state_file:
            pyboy.load_state(state_file)
        print(f"State carregado: {state_path}")
    else:
        print(f"State nao encontrado, iniciando da ROM: {state_path}")

    reader = PokemonRedRamReader(pyboy)
    stepper = StepHandler(pyboy, action_frames=args.action_frames, render=args.window != "null")

    print("Comandos: w/a/s/d mover, j=A, k=B, u=START, i=SELECT, .=noop")
    print("p=print RAM, save [arquivo.state], q=sair")

    try:
        print_snapshot(reader)
        while True:
            command = input("> ").strip()
            if command == "q":
                break
            if command == "p":
                print_snapshot(reader)
                continue
            if command.startswith("save"):
                _, _, raw_path = command.partition(" ")
                output = Path(raw_path.strip()) if raw_path.strip() else state_path
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("wb") as state_file:
                    pyboy.save_state(state_file)
                print(f"State salvo: {output}")
                print_snapshot(reader)
                continue

            action_name = KEY_TO_ACTION.get(command)
            if action_name is None:
                print("Comando desconhecido.")
                continue

            stepper.run(ACTIONS.index(action_name))
            print_snapshot(reader)
    finally:
        pyboy.stop(save=False)


def print_snapshot(reader: PokemonRedRamReader) -> None:
    snapshot = reader.snapshot()
    print(
        f"map={snapshot.map_id} x={snapshot.x} y={snapshot.y} "
        f"battle={int(snapshot.in_battle)} hp={snapshot.hp_fraction:.2f} "
        f"badges={snapshot.badge_count} events={snapshot.event_count} "
        f"party={snapshot.party_count} levels={list(snapshot.party_levels)}"
    )


if __name__ == "__main__":
    main()
