from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import PokemonRedRamReader, detect_dialog_from_pyboy
from phases import get_phase
from project_config import env_int, env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Monitora se a tela esta em dialogo/textbox.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--state", default=None)
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--window", default="SDL2")
    parser.add_argument("--seconds", type=int, default=0, help="0 roda ate Ctrl+C.")
    parser.add_argument("--print-every", type=int, default=env_int("POKE_DIALOG_PROBE_EVERY", 30))
    parser.add_argument("--screenshot", default=None, help="Salva screenshot final nesse caminho.")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    state_path = Path(args.state) if args.state else Path(args.states_dir) / phase.state

    from pyboy import PyBoy

    pyboy = PyBoy(args.rom, window=args.window)
    reader = PokemonRedRamReader(pyboy)
    try:
        if state_path.exists():
            with state_path.open("rb") as state_file:
                pyboy.load_state(state_file)
            print(f"State carregado: {state_path}")
        else:
            print(f"State nao encontrado; rodando direto da ROM: {state_path}")

        print("Monitore a janela. Use Ctrl+C no terminal para parar.")
        frame_limit = None if args.seconds <= 0 else args.seconds * 60
        frame = 0
        while frame_limit is None or frame < frame_limit:
            running = pyboy.tick()
            if not running:
                break
            frame += 1

            if frame % max(1, args.print_every) == 0:
                snapshot = reader.snapshot()
                detection = detect_dialog_from_pyboy(pyboy)
                print(
                    f"frame={frame} "
                    f"dialog_visual={detection.visual} "
                    f"score={detection.score:.3f} "
                    f"white={detection.bottom_white:.3f} "
                    f"dark={detection.bottom_dark:.3f} "
                    f"wx={detection.window_x} wy={detection.window_y} "
                    f"win={detection.window_on_screen} "
                    f"tiles={detection.bottom_unique_tiles} "
                    f"map={snapshot.map_id} x={snapshot.x} y={snapshot.y} "
                    f"events={snapshot.event_count} battle={int(snapshot.in_battle)}"
                )
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nParado pelo usuario.")
    finally:
        if args.screenshot:
            save_screenshot(pyboy, Path(args.screenshot))
        pyboy.stop(save=False)


def save_screenshot(pyboy, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pyboy.screen.image.save(output)
    print(f"Screenshot salva: {output}")


if __name__ == "__main__":
    main()
