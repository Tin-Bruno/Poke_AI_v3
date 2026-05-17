from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import PokemonRedRamReader
from phases import get_phase
from project_config import env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Mostra RAM basica de uma fase/state.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--state", default=None)
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--window", default="null")
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
    try:
        with state_path.open("rb") as state_file:
            pyboy.load_state(state_file)
        snapshot = PokemonRedRamReader(pyboy).snapshot()
        print(snapshot.to_info())
    finally:
        pyboy.stop(save=False)


if __name__ == "__main__":
    main()
