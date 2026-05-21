from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume um trace CSV gerado pelo eval_freeplay.")
    parser.add_argument("trace_csv")
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.trace_csv)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"Trace vazio: {path}")

    maps = Counter(row["map_id"] for row in rows)
    coords = {(row["map_id"], row["x"], row["y"]) for row in rows}
    actions = Counter(row["action"] for row in rows)
    last = rows[-1]

    print("")
    print("=== Trace freeplay ===")
    print(f"Arquivo:       {path}")
    print(f"Steps:         {len(rows)}")
    print(f"Mapas unicos:  {len(maps)}")
    print(f"Coords unicas: {len(coords)}")
    print(f"Final:         map={last.get('map_id')} x={last.get('x')} y={last.get('y')}")
    print(f"Eventos:       {last.get('event_count')}")
    print(f"Batalhas:      {last.get('battles_started')} iniciadas, {last.get('battles_won')} vencidas")
    print("")
    print("Mapas mais visitados:")
    for map_id, count in maps.most_common(args.top):
        print(f"  map={map_id}: {count} steps")
    print("")
    print("Acoes mais usadas:")
    for action, count in actions.most_common(args.top):
        print(f"  {action}: {count}")
    print("=====================")


if __name__ == "__main__":
    main()
