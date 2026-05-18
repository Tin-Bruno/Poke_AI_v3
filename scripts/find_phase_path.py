from __future__ import annotations

import argparse
import sys
from collections import deque
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import PokemonRedRamReader
from phases import get_phase
from project_config import env_int, env_str, load_dotenv


DIRECTIONS = ("up", "down", "left", "right")
MANUAL_KEYS = {
    "up": "w",
    "down": "s",
    "left": "a",
    "right": "d",
}


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Procura uma rota simples ate o mapa alvo.")
    parser.add_argument("--rom", default=env_str("POKE_ROM_PATH"))
    parser.add_argument("--phase", default=env_str("POKE_PHASE", "phase1"))
    parser.add_argument("--states-dir", default=env_str("POKE_STATES_DIR", "states"))
    parser.add_argument("--symbols", default=env_str("POKE_SYMBOLS_PATH"))
    parser.add_argument("--target-map", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--action-frames", type=int, default=env_int("POKE_ACTION_FRAMES", 12))
    parser.add_argument("--wait-frames", type=int, default=env_int("POKE_MANUAL_WAIT_FRAMES", 12))
    args = parser.parse_args()
    if not args.rom:
        parser.error("informe --rom ou POKE_ROM_PATH no .env")
    return args


def main() -> None:
    args = parse_args()
    phase = get_phase(args.phase)
    target_map = phase.target_map if args.target_map is None else args.target_map
    if target_map is None:
        raise ValueError("A fase nao tem target_map. Informe --target-map.")

    state_path = Path(args.states_dir) / phase.state
    if not state_path.exists():
        raise FileNotFoundError(f"State nao encontrado: {state_path}")

    from pyboy import PyBoy

    kwargs = {"window": "null", "sound_emulated": False}
    if args.symbols:
        kwargs["symbols"] = args.symbols

    pyboy = PyBoy(args.rom, **kwargs)
    pyboy.set_emulation_speed(0)
    reader = PokemonRedRamReader(pyboy)

    try:
        with state_path.open("rb") as state_file:
            pyboy.load_state(state_file)
        tick_many(pyboy, 30)

        start_state = dump_state(pyboy)
        start_position = position(reader)
        queue = deque([(start_state, [])])
        seen = {start_position}
        found: tuple[tuple[int, int, int], list[str]] | None = None

        while queue:
            state, path = queue.popleft()
            if len(path) >= args.max_depth:
                continue

            for direction in DIRECTIONS:
                load_state(pyboy, state)
                tap(pyboy, direction, args.action_frames, args.wait_frames)
                current_position = position(reader)
                next_path = [*path, direction]

                if current_position[0] == target_map:
                    found = (current_position, next_path)
                    queue.clear()
                    break

                if current_position not in seen:
                    seen.add(current_position)
                    queue.append((dump_state(pyboy), next_path))

            if found:
                break

        print(f"inicio: map={start_position[0]} x={start_position[1]} y={start_position[2]}")
        print(f"posicoes_visitadas: {len(seen)}")
        if not found:
            seen_maps = sorted({map_id for map_id, _, _ in seen})
            print(f"mapas_visitados: {seen_maps}")
            print(f"rota_nao_encontrada ate map={target_map} em {args.max_depth} movimentos")
            return

        final_position, path = found
        print(f"alvo: map={final_position[0]} x={final_position[1]} y={final_position[2]}")
        print(f"movimentos: {' '.join(path)}")
        print(f"manual_control: {''.join(MANUAL_KEYS[action] for action in path)}")
    finally:
        pyboy.stop(save=False)


def tick_many(pyboy, frames: int) -> bool:
    for _ in range(max(0, frames)):
        if not bool(pyboy.tick(1, False, False)):
            return False
    return True


def tap(pyboy, button: str, action_frames: int, wait_frames: int) -> None:
    pyboy.button_press(button)
    tick_many(pyboy, action_frames)
    pyboy.button_release(button)
    tick_many(pyboy, wait_frames)


def dump_state(pyboy) -> bytes:
    state_file = BytesIO()
    pyboy.save_state(state_file)
    return state_file.getvalue()


def load_state(pyboy, state: bytes) -> None:
    pyboy.load_state(BytesIO(state))


def position(reader: PokemonRedRamReader) -> tuple[int, int, int]:
    snapshot = reader.snapshot()
    return snapshot.map_id, snapshot.x, snapshot.y


if __name__ == "__main__":
    main()
