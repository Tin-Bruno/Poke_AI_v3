"""Leitura de progresso do jogo pela RAM do Pokemon Red/Blue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FALLBACK_SYMBOLS: dict[str, int] = {
    "wObtainedBadges": 0xD356,
    "wCurMap": 0xD35E,
    "wYCoord": 0xD361,
    "wXCoord": 0xD362,
    "wPartyCount": 0xD163,
    "wIsInBattle": 0xD057,
}

PARTY_LEVEL_FALLBACKS: tuple[int, ...] = tuple(0xD18C + (44 * index) for index in range(6))
PARTY_SPECIES_FALLBACKS: tuple[int, ...] = tuple(0xD164 + index for index in range(6))
PARTY_HP_FALLBACKS: tuple[int, ...] = tuple(0xD16C + (44 * index) for index in range(6))
PARTY_MAX_HP_FALLBACKS: tuple[int, ...] = tuple(0xD18D + (44 * index) for index in range(6))
OPPONENT_LEVEL_FALLBACKS: tuple[int, ...] = tuple(0xD8C5 + (44 * index) for index in range(6))

EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
EVENT_BITS = (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8


@dataclass(frozen=True)
class GameSnapshot:
    """Pequeno resumo do estado atual que alimenta a recompensa."""

    map_id: int
    x: int
    y: int
    badges: int
    party_count: int
    party_levels: tuple[int, ...]
    party_species: tuple[int, ...] = ()
    hp_fraction: float = 0.0
    in_battle: bool = False
    event_count: int = 0
    event_bits: tuple[int, ...] = ()
    max_opponent_level: int = 0

    @property
    def badge_count(self) -> int:
        return int(self.badges).bit_count()

    @property
    def max_party_level(self) -> int:
        return max(self.party_levels, default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "x": self.x,
            "y": self.y,
            "badges": self.badges,
            "badge_count": self.badge_count,
            "party_count": self.party_count,
            "party_levels": self.party_levels,
            "party_species": self.party_species,
            "hp_fraction": self.hp_fraction,
            "in_battle": self.in_battle,
            "event_count": self.event_count,
            "max_opponent_level": self.max_opponent_level,
        }


class PokemonRedMemoryReader:
    """Le Pokemon Red/Blue usando simbolos do PyBoy ou enderecos vanilla."""

    def __init__(self, pyboy: Any) -> None:
        self.pyboy = pyboy
        self._symbol_cache: dict[str, Any] = {}

    def snapshot(self) -> GameSnapshot:
        party_count = max(0, min(6, self._read("wPartyCount")))
        party_levels = tuple(self._read_party_level(index) for index in range(party_count))
        party_species = tuple(self._read_party_species(index) for index in range(party_count))
        event_bits = self._read_event_bits()

        return GameSnapshot(
            map_id=self._read("wCurMap"),
            x=self._read("wXCoord"),
            y=self._read("wYCoord"),
            badges=self._read("wObtainedBadges"),
            party_count=party_count,
            party_levels=party_levels,
            party_species=party_species,
            hp_fraction=self._read_hp_fraction(party_count),
            in_battle=self._read("wIsInBattle") != 0,
            event_count=sum(event_bits),
            event_bits=event_bits,
            max_opponent_level=self._read_max_opponent_level(),
        )

    def _read(self, symbol: str) -> int:
        address = self._resolve(symbol)
        return int(self.pyboy.memory[address])

    def _read_party_level(self, index: int) -> int:
        symbol = f"wPartyMon{index + 1}Level"
        try:
            return self._read(symbol)
        except Exception:
            return int(self.pyboy.memory[PARTY_LEVEL_FALLBACKS[index]])

    def _read_party_species(self, index: int) -> int:
        return int(self.pyboy.memory[PARTY_SPECIES_FALLBACKS[index]])

    def _read_event_bits(self) -> tuple[int, ...]:
        return tuple(
            int(bit)
            for address in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            for bit in f"{int(self.pyboy.memory[address]):08b}"
        )

    def _read_hp_fraction(self, party_count: int) -> float:
        if party_count <= 0:
            return 0.0

        hp_sum = sum(self._read_u16(address) for address in PARTY_HP_FALLBACKS[:party_count])
        max_hp_sum = sum(self._read_u16(address) for address in PARTY_MAX_HP_FALLBACKS[:party_count])
        return hp_sum / max(max_hp_sum, 1)

    def _read_max_opponent_level(self) -> int:
        return max((int(self.pyboy.memory[address]) for address in OPPONENT_LEVEL_FALLBACKS), default=0)

    def _read_u16(self, address: int) -> int:
        return 256 * int(self.pyboy.memory[address]) + int(self.pyboy.memory[address + 1])

    def _resolve(self, symbol: str) -> Any:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        try:
            address = self.pyboy.symbol_lookup(symbol)
        except Exception:
            address = FALLBACK_SYMBOLS[symbol]

        self._symbol_cache[symbol] = address
        return address
