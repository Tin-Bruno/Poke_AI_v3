from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.step_handler import ACTIONS, StepHandler
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
    parser = argparse.ArgumentParser(
        description="Testa a ROM pura no PyBoy, sem carregar state e sem ambiente Gym."
    )
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"), help="Caminho da ROM.")
    parser.add_argument("--window", default="SDL2", help='Use "SDL2" para janela ou "null".')
    parser.add_argument(
        "--mode",
        choices=("native", "scripted"),
        default="native",
        help="native joga direto na janela PyBoy; scripted manda botoes pelo terminal.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=180,
        help="Tempo do modo native. Use 0 para rodar ate Ctrl+C.",
    )
    parser.add_argument(
        "--action-frames",
        type=int,
        default=env_int("POKE_ROM_TEST_ACTION_FRAMES", 12),
    )
    parser.add_argument(
        "--wait-frames",
        type=int,
        default=env_int("POKE_ROM_TEST_WAIT_FRAMES", 12),
    )
    parser.add_argument(
        "--dialog-wait-frames",
        type=int,
        default=env_int("POKE_ROM_TEST_DIALOG_WAIT_FRAMES", 60),
    )
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    rom_path = Path(args.rom)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM nao encontrada: {rom_path}")

    print_rom_info(rom_path)
    warn_if_sav_exists(rom_path)

    from pyboy import PyBoy

    kwargs = {"window": args.window}
    if args.window == "null":
        kwargs["sound_emulated"] = False

    pyboy = PyBoy(str(rom_path), **kwargs)
    try:
        if args.mode == "native":
            run_native(pyboy, args.seconds)
        else:
            pyboy.set_emulation_speed(0)
            stepper = StepHandler(
                pyboy,
                action_frames=args.action_frames,
                wait_frames=args.wait_frames,
                render=args.window != "null",
            )
            run_scripted(pyboy, stepper, args)
    finally:
        pyboy.stop(save=False)


def run_native(pyboy, seconds: int) -> None:
    pyboy.set_emulation_speed(1)
    frame_limit = None if seconds <= 0 else seconds * 60
    print("")
    print("Modo native: clique na janela do PyBoy e jogue usando o teclado do PyBoy.")
    print("Este modo nao carrega state e nao usa nosso manual_control/env.")
    print("Use Ctrl+C no terminal para encerrar.")
    print("")

    frames = 0
    try:
        while frame_limit is None or frames < frame_limit:
            running = pyboy.tick()
            if not running:
                break
            frames += 1
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuario.")


def run_scripted(pyboy, stepper: StepHandler, args: argparse.Namespace) -> None:
    print("")
    print("Modo scripted: sem state, botoes pelo terminal.")
    print("Comandos: j=A, k=B, u=START, i=SELECT, w/a/s/d mover")
    print("Sequencias: jjjj, 10j, ddddww")
    print("Dialogo: talk 10 ou t 10")
    print("Outros: wait 60, save logs/rom_test.state, q")
    print("")

    while True:
        command = input("rom> ").strip().lower()
        if command == "q":
            break
        if command.startswith("save"):
            _, _, raw_path = command.partition(" ")
            output = Path(raw_path.strip()) if raw_path.strip() else Path("logs/rom_test.state")
            save_state(pyboy, output)
            print(f"State salvo: {output}")
            continue
        if command.startswith("talk") or command.startswith("t "):
            count = parse_count_command(command, default=10)
            run_dialog_mash(stepper, count, args.dialog_wait_frames)
            print(f"{count} A de dialogo executados.")
            continue
        if command.startswith("wait"):
            frames = parse_count_command(command, default=60)
            stepper.wait(frames)
            print(f"{frames} frames aguardados.")
            continue

        actions = expand_action_sequence(command)
        if not actions:
            print("Comando desconhecido.")
            continue

        for action_name in actions:
            stepper.run(ACTIONS.index(action_name))
        print(f"{len(actions)} acoes executadas.")


def print_rom_info(rom_path: Path) -> None:
    data = rom_path.read_bytes()
    title = data[0x134:0x144].split(b"\x00", 1)[0].decode("ascii", "replace")
    header_checksum = data[0x14D] if len(data) > 0x14D else None
    global_checksum = data[0x14E:0x150].hex() if len(data) > 0x150 else "?"
    print(f"ROM: {rom_path}")
    print(f"Tamanho: {len(data)} bytes")
    print(f"Titulo header: {title!r}")
    print(f"Header checksum: {header_checksum!r}")
    print(f"Global checksum: {global_checksum}")


def warn_if_sav_exists(rom_path: Path) -> None:
    sav_path = rom_path.with_suffix(".sav")
    if sav_path.exists():
        print("")
        print(f"Aviso: existe SRAM ao lado da ROM: {sav_path}")
        print("Se quiser testar totalmente limpo, mova esse .sav temporariamente.")


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


if __name__ == "__main__":
    main()
