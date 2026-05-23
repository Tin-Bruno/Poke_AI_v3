from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.freeplay_env import FreeplayConfig, FreeplayPokemonRedEnv
from project_config import env_str, load_dotenv
from rewards.freeplay_reward import FreeplayReward


REWARD_TRACE_FIELDS = (
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

TRACE_FIELDNAMES = (
    "step",
    "map_id",
    "x",
    "y",
    "reward",
    "action",
    "event_count",
    "party_count",
    "party_levels",
    "hp_fraction",
    "in_battle",
    "seen_maps",
    "seen_locations",
    "seen_coords",
    "coord_visit_count",
    "steps_since_progress",
    "freeplay_progress",
    "battles_started",
    "battles_won",
    "best_seen_maps",
    "best_seen_locations",
    "best_event_count",
    "best_party_level_total",
    "blocked_move_penalty",
    *REWARD_TRACE_FIELDS,
)


def should_stop_eval() -> bool:
    try:
        import msvcrt
    except ImportError:
        return False

    if not msvcrt.kbhit():
        return False

    key = msvcrt.getch()
    if key in (b"\x00", b"\xe0"):
        if msvcrt.kbhit():
            msvcrt.getch()
        return False

    return key.decode("utf-8", errors="ignore").lower() == "q"


def interruptible_delay(seconds: float) -> bool:
    if seconds <= 0:
        return False
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        if should_stop_eval():
            return True
        time.sleep(min(0.05, max(0.0, end_time - time.monotonic())))
    return False


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Avalia um modelo freeplay.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--state", default=env_str("POKE_FREEPLAY_STATE", "states/freeplay_start.state"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--model", default="models/pokemon_red_freeplay.zip")
    parser.add_argument("--window", choices=("null", "SDL2"), default="SDL2")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--action-frames", type=int, default=24)
    parser.add_argument("--release-frames", type=int, default=6)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--stagnation-steps", type=int, default=4_000)
    parser.add_argument("--action-set", choices=("simple", "combo"), default="simple")
    parser.add_argument("--observation-mode", choices=("coords", "ram", "multi"), default="multi")
    parser.add_argument("--legacy-observation", action="store_true")
    parser.add_argument("--visited-radius", type=int, default=12)
    parser.add_argument("--blocked-move-penalty", type=float, default=0.02)
    parser.add_argument("--same-coord-stuck-steps", type=int, default=600)
    parser.add_argument("--same-coord-stuck-penalty", type=float, default=0.05)
    parser.add_argument("--emulation-speed", type=float, default=None)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--trace-csv", default=None)
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    config = FreeplayConfig(
        rom_path=args.rom,
        state_path=args.state,
        symbols_path=args.symbols,
        window=args.window,
        action_frames=args.action_frames,
        release_frames=args.release_frames,
        warmup_frames=args.warmup_frames,
        max_steps=args.max_steps,
        stagnation_steps=args.stagnation_steps,
        action_set=args.action_set,
        observation_mode=args.observation_mode,
        memory_observation=not args.legacy_observation,
        visited_radius=args.visited_radius,
        blocked_move_penalty=args.blocked_move_penalty,
        emulation_speed=args.emulation_speed,
    )
    reward_model = FreeplayReward(
        same_coord_limit=args.same_coord_stuck_steps,
        same_coord_penalty=args.same_coord_stuck_penalty,
    )
    env = FreeplayPokemonRedEnv(config, reward_model=reward_model)
    trace_file = None
    try:
        model = PPO.load(args.model)
        if model.action_space != env.action_space or model.observation_space != env.observation_space:
            raise SystemExit(
                "Modelo freeplay incompativel com a configuracao atual. "
                "Para avaliar o modelo antigo, use --action-set combo --legacy-observation. "
                "Para a logica nova, treine novamente com scripts/train_freeplay.py."
            )

        obs, info = env.reset()
        final_info = info
        total_reward = 0.0
        stop_reason = "limite de steps"
        trace_writer = None
        if args.trace_csv:
            trace_path = Path(args.trace_csv)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = trace_path.open("w", newline="", encoding="utf-8")
            trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDNAMES)
            trace_writer.writeheader()

        for _ in range(args.steps):
            if should_stop_eval():
                stop_reason = "interrompido com q"
                break
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            final_info = info
            if trace_writer:
                trace_writer.writerow(trace_row(info, reward))
            if terminated or truncated:
                stop_reason = "episodio encerrado"
                break
            if interruptible_delay(args.delay):
                stop_reason = "interrompido com q"
                break

        print_summary(final_info, total_reward, stop_reason)
    finally:
        if trace_file:
            trace_file.close()
        env.close()


def print_summary(info: dict, total_reward: float, stop_reason: str) -> None:
    print("")
    print("=== Freeplay final ===")
    print(f"Motivo:      {stop_reason}")
    print(f"Passos:      {info.get('step_count')}")
    print(f"Posicao:     map={info.get('map_id')} x={info.get('x')} y={info.get('y')}")
    print(f"Acao final:  {info.get('action')}")
    print(f"Reward:      {total_reward:.3f}")
    print(f"Mapas:       {int(info.get('seen_maps', 0))}")
    print(f"Locais:      {int(info.get('seen_locations', 0))}")
    print(f"Coords:      {int(info.get('seen_coords', 0))}")
    print(f"Eventos:     {int(info.get('event_count', 0))}")
    print(f"Batalhas:    {info.get('battles_started', 0)} iniciadas, {info.get('battles_won', 0)} vencidas")
    print(f"Party:       {info.get('party_count')} levels={info.get('party_levels')}")
    print("======================")


def trace_row(info: dict, reward: float) -> dict:
    row = {
        "step": info.get("step_count"),
        "map_id": info.get("map_id"),
        "x": info.get("x"),
        "y": info.get("y"),
        "reward": float(reward),
        "action": info.get("action"),
        "event_count": info.get("event_count"),
        "party_count": info.get("party_count"),
        "party_levels": info.get("party_levels"),
        "hp_fraction": info.get("hp_fraction"),
        "in_battle": info.get("in_battle"),
        "seen_maps": info.get("seen_maps", info.get("best_seen_maps")),
        "seen_locations": info.get("seen_locations", info.get("best_seen_locations")),
        "seen_coords": info.get("seen_coords"),
        "coord_visit_count": info.get("coord_visit_count"),
        "steps_since_progress": info.get("steps_since_progress"),
        "freeplay_progress": info.get("freeplay_progress"),
        "battles_started": info.get("battles_started"),
        "battles_won": info.get("battles_won"),
        "best_seen_maps": info.get("best_seen_maps"),
        "best_seen_locations": info.get("best_seen_locations"),
        "best_event_count": info.get("best_event_count"),
        "best_party_level_total": info.get("best_party_level_total"),
        "blocked_move_penalty": info.get("blocked_move_penalty"),
    }
    row.update({field: info.get(field, 0.0) for field in REWARD_TRACE_FIELDS})
    return row


if __name__ == "__main__":
    main()
