from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phases import PHASES


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumo visual simples das fases configuradas.")
    parser.parse_args()
    for phase_id, phase in PHASES.items():
        print(
            f"{phase_id:>6} | {phase.name:<32} | state={phase.state:<24} "
            f"| success={phase.success}"
        )


if __name__ == "__main__":
    main()
