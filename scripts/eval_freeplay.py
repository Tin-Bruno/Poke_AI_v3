from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.freeplay_env import FreeplayConfig, FreeplayPokemonRedEnv
from project_config import env_str, load_dotenv


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
    parser.add_argument("--state", default="states/phase9_start.state")
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--model", default="models/pokemon_red_freeplay.zip")
    parser.add_argument("--window", choices=("null", "SDL2"), default="SDL2")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--action-frames", type=int, default=24)
    parser.add_argument("--release-frames", type=int, default=6)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--stagnation-steps", type=int, default=4_000)
    parser.add_argument("--emulation-speed", type=float, default=None)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
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
        emulation_speed=args.emulation_speed,
    )
    env = FreeplayPokemonRedEnv(config)
    try:
        model = PPO.load(args.model)
        if model.action_space != env.action_space or model.observation_space != env.observation_space:
            raise SystemExit("Modelo freeplay incompativel. Treine novamente com scripts/train_freeplay.py.")

        obs, info = env.reset()
        final_info = info
        total_reward = 0.0
        stop_reason = "limite de steps"

        for _ in range(args.steps):
            if should_stop_eval():
                stop_reason = "interrompido com q"
                break
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            final_info = info
            if terminated or truncated:
                stop_reason = "episodio encerrado"
                break
            if interruptible_delay(args.delay):
                stop_reason = "interrompido com q"
                break

        print_summary(final_info, total_reward, stop_reason)
    finally:
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
    print(f"Eventos:     {int(info.get('event_count', 0))}")
    print(f"Party:       {info.get('party_count')} levels={info.get('party_levels')}")
    print("======================")


if __name__ == "__main__":
    main()
