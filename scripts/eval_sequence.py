from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda varias fases em sequencia.")
    parser.add_argument("phases", nargs="+", help="Ex: phase1 phase2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Por enquanto rode as fases uma a uma para salvar/conferir states entre elas:")
    for phase in args.phases:
        print(f"python scripts/eval_phase.py --phase {phase}")


if __name__ == "__main__":
    main()
