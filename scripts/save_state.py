from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phases import get_phase
from project_config import env_str, load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Salva um state jogando direto na janela PyBoy, sem injetar botoes."
    )
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"), help="Caminho da ROM.")
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"), help="Fase usada no output.")
    parser.add_argument("--load-state", default=None, help="State opcional para carregar antes de jogar.")
    parser.add_argument("--output", default=None, help="State de saida. Default: state da fase.")
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--window", default="SDL2")
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    output = Path(args.output) if args.output else Path(args.states_dir) / phase.state

    from pyboy import PyBoy

    kwargs = {"window": args.window}
    if args.window == "null":
        kwargs["sound_emulated"] = False

    pyboy = PyBoy(args.rom, **kwargs)
    try:
        if args.load_state:
            with Path(args.load_state).open("rb") as state_file:
                pyboy.load_state(state_file)
            print(f"State carregado: {args.load_state}")

        print("Jogue diretamente na janela do PyBoy.")
        print("Quando estiver no ponto desejado, feche a janela do PyBoy.")
        print(f"O state sera salvo em: {output}")

        while pyboy.tick():
            pass

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as state_file:
            pyboy.save_state(state_file)
        print(f"State salvo: {output}")
    finally:
        pyboy.stop(save=False)


if __name__ == "__main__":
    main()
