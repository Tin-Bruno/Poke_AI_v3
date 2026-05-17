from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

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


def get_single_key_reader() -> Callable[[], str] | None:
    try:
        import msvcrt
    except ImportError:
        return None

    def read_key() -> str:
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            return ""
        return key.lower()

    return read_key


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Controle manual para criar states e inspecionar RAM.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"), help="Caminho da ROM.")
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"), help="Fase base.")
    parser.add_argument("--state", default=None, help="State inicial opcional.")
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default=env_str("POKE_WINDOW", "SDL2"))
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_MANUAL_ACTION_FRAMES", 12))
    parser.add_argument("--wait-frames", type=int, default=env_int("POKE_MANUAL_WAIT_FRAMES", 12))
    parser.add_argument("--dialog-wait-frames", type=int, default=env_int("POKE_MANUAL_DIALOG_WAIT_FRAMES", 45))
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Le teclas sem Enter. No Git Bash, prefira o modo padrao com sequencias.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=env_int("POKE_MANUAL_PRINT_EVERY", 0),
        help="Imprime RAM a cada N acoes. Use 0 para imprimir so com p/save.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Caminho usado pela tecla v no modo rapido.",
    )
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    state_path = Path(args.state) if args.state else Path(args.states_dir) / phase.state

    from pyboy import PyBoy

    kwargs = {"window": args.window}
    if args.window == "null":
        kwargs["sound_emulated"] = False
    if args.symbols:
        kwargs["symbols"] = args.symbols

    pyboy = PyBoy(args.rom, **kwargs)
    pyboy.set_emulation_speed(0 if args.window == "null" else 1)
    if state_path.exists():
        with state_path.open("rb") as state_file:
            pyboy.load_state(state_file)
        print(f"State carregado: {state_path}")
    else:
        print(f"State nao encontrado, iniciando da ROM: {state_path}")

    reader = PokemonRedRamReader(pyboy)
    stepper = StepHandler(
        pyboy,
        action_frames=args.action_frames,
        wait_frames=args.wait_frames,
        render=args.window != "null",
    )

    print("Comandos: w/a/s/d mover, j=A, k=B, u=START, i=SELECT, .=noop")
    print("Digite sequencias e aperte Enter. Exemplos: jjjj, ddddww, 10j, 5d")
    print("dialogo: talk 10 ou t 10 aperta A com pausa maior")
    print("p=RAM, save [arquivo.state]=salvar, q=sair")

    try:
        print_snapshot(reader)
        if args.fast_mode:
            read_key = get_single_key_reader()
            if read_key is None:
                print("Modo rapido indisponivel neste terminal. Usando modo de sequencias.")
                run_sequence_mode(pyboy, reader, stepper, state_path, args)
            else:
                save_path = Path(args.save_path) if args.save_path else state_path
                run_fast_mode(pyboy, reader, stepper, save_path, args, read_key)
        else:
            run_sequence_mode(pyboy, reader, stepper, state_path, args)
    finally:
        pyboy.stop(save=False)


def run_fast_mode(
    pyboy,
    reader: PokemonRedRamReader,
    stepper: StepHandler,
    save_path: Path,
    args: argparse.Namespace,
    read_key: Callable[[], str],
) -> None:
    action_count = 0
    while True:
        key = read_key()
        if not key:
            continue
        if key == "q":
            print("")
            break
        if key == "p":
            print("")
            print_snapshot(reader)
            continue
        if key == "v":
            save_state(pyboy, save_path)
            print("")
            print(f"State salvo: {save_path}")
            print_snapshot(reader)
            continue

        action_name = KEY_TO_ACTION.get(key)
        if action_name is None:
            continue

        stepper.run(ACTIONS.index(action_name))
        action_count += 1
        if args.print_every and action_count % args.print_every == 0:
            print("")
            print_snapshot(reader)
        elif action_count % 25 == 0:
            print(".", end="", flush=True)


def run_sequence_mode(
    pyboy,
    reader: PokemonRedRamReader,
    stepper: StepHandler,
    state_path: Path,
    args: argparse.Namespace,
) -> None:
    action_count = 0
    while True:
        command = input("> ").strip().lower()
        if command == "q":
            break
        if command == "p":
            print_snapshot(reader)
            continue
        if command.startswith("save"):
            _, _, raw_path = command.partition(" ")
            output = Path(raw_path.strip()) if raw_path.strip() else state_path
            save_state(pyboy, output)
            print(f"State salvo: {output}")
            print_snapshot(reader)
            continue
        if command.startswith("talk") or command.startswith("t "):
            count = parse_count_command(command, default=10)
            run_dialog_mash(stepper, count, args.dialog_wait_frames)
            action_count += count
            print(f"{count} A de dialogo executados.")
            continue
        if command.startswith("wait"):
            count = parse_count_command(command, default=60)
            stepper.wait(count)
            print(f"{count} frames aguardados.")
            continue

        actions = expand_action_sequence(command)
        if not actions:
            print("Comando desconhecido.")
            continue

        for action_name in actions:
            stepper.run(ACTIONS.index(action_name))
            action_count += 1
            if args.print_every and action_count % args.print_every == 0:
                print_snapshot(reader)
        if not args.print_every:
            print(f"{len(actions)} acoes executadas.")


def expand_action_sequence(command: str) -> list[str]:
    actions: list[str] = []
    repeat_buffer = ""
    for char in command.replace(" ", ""):
        if char.isdigit():
            repeat_buffer += char
            continue

        action_name = KEY_TO_ACTION.get(char)
        if action_name is None:
            return []

        repeat = int(repeat_buffer) if repeat_buffer else 1
        repeat_buffer = ""
        actions.extend([action_name] * repeat)

    if repeat_buffer:
        return []
    return actions


def parse_count_command(command: str, default: int) -> int:
    parts = command.split()
    if len(parts) < 2:
        return default
    try:
        return max(1, int(parts[1]))
    except ValueError:
        return default


def run_dialog_mash(stepper: StepHandler, count: int, dialog_wait_frames: int) -> None:
    for _ in range(count):
        stepper.run(ACTIONS.index("a"))
        stepper.wait(dialog_wait_frames)


def save_state(pyboy, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as state_file:
        pyboy.save_state(state_file)


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
