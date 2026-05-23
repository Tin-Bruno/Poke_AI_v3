from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


REWARD_FIELDS = (
    "reward_coord",
    "reward_location",
    "reward_map",
    "reward_events",
    "reward_badges",
    "reward_party",
    "reward_survival",
    "reward_battle",
    "reward_repeat_penalty",
    "reward_stuck_penalty",
    "reward_faint_penalty",
    "reward_step_penalty",
    "reward_blocked_move_penalty",
)


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
    action_rewards: Counter[str] = Counter()
    for row in rows:
        action_rewards[row["action"]] += as_float(row, "reward")
    last = rows[-1]
    reward_total = sum(as_float(row, "reward") for row in rows)
    progress_steps = sum(1 for row in rows if as_float(row, "freeplay_progress") > 0.0)
    max_stall = max((as_float(row, "steps_since_progress") for row in rows), default=0.0)
    reward_totals = {
        field: sum(as_float(row, field) for row in rows)
        for field in REWARD_FIELDS
        if field in rows[0]
    }

    print("")
    print("=== Trace freeplay ===")
    print(f"Arquivo:       {path}")
    print(f"Steps:         {len(rows)}")
    print(f"Mapas unicos:  {len(maps)}")
    print(f"Coords unicas: {len(coords)}")
    print(f"Final:         map={last.get('map_id')} x={last.get('x')} y={last.get('y')}")
    print(f"Eventos:       {last.get('event_count')}")
    print(f"Batalhas:      {last.get('battles_started')} iniciadas, {last.get('battles_won')} vencidas")
    print(f"Reward total:  {reward_total:.3f}")
    if "freeplay_progress" in rows[0]:
        print(f"Progresso:     {progress_steps} steps")
    if "steps_since_progress" in rows[0]:
        print(f"Maior trava:   {max_stall:.0f} steps sem progresso")
    print("")
    print("Mapas mais visitados:")
    for map_id, count in maps.most_common(args.top):
        print(f"  map={map_id}: {count} steps")
    print("")
    print("Acoes mais usadas:")
    for action, count in actions.most_common(args.top):
        average_reward = action_rewards[action] / max(count, 1)
        print(f"  {action}: {count} | reward medio={average_reward:.4f}")
    if reward_totals:
        print("")
        print("Reward por termo:")
        for field, value in sorted(reward_totals.items(), key=lambda item: abs(item[1]), reverse=True):
            if abs(value) < 1e-9:
                continue
            print(f"  {field}: {value:.3f}")
    print("=====================")


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


if __name__ == "__main__":
    main()
