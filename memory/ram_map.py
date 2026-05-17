"""Mapa de RAM e snapshot de progresso para Pokemon Red/Blue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RAM_SYMBOLS: dict[str, int] = {
    "wIsInBattle": 0xD057,
    "wPartyCount": 0xD163,
    "wObtainedBadges": 0xD356,
    "wCurMap": 0xD35E,
    "wYCoord": 0xD361,
    "wXCoord": 0xD362,
}

PARTY_SPECIES = tuple(0xD164 + index for index in range(6))
PARTY_HP = tuple(0xD16C + (44 * index) for index in range(6))
PARTY_LEVELS = tuple(0xD18C + (44 * index) for index in range(6))
PARTY_MAX_HP = tuple(0xD18D + (44 * index) for index in range(6))
OPPONENT_LEVELS = tuple(0xD8C5 + (44 * index) for index in range(6))

EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E
EVENT_BITS = (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8


@dataclass(frozen=True)
class GameSnapshot:
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

    @property
    def position(self) -> tuple[int, int, int]:
        return self.map_id, self.y, self.x

    def to_info(self) -> dict[str, int | float | tuple[int, ...] | bool]:
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


class PokemonRedRamReader:
    """Le memoria do Pokemon Red/Blue usando simbolos quando disponiveis."""

    def __init__(self, pyboy: Any) -> None:
        self.pyboy = pyboy
        self._cache: dict[str, Any] = {}

    def snapshot(self) -> GameSnapshot:
        party_count = max(0, min(6, self.read_symbol("wPartyCount")))
        event_bits = self.read_event_bits()
        return GameSnapshot(
            map_id=self.read_symbol("wCurMap"),
            x=self.read_symbol("wXCoord"),
            y=self.read_symbol("wYCoord"),
            badges=self.read_symbol("wObtainedBadges"),
            party_count=party_count,
            party_levels=tuple(self.read(PARTY_LEVELS[index]) for index in range(party_count)),
            party_species=tuple(self.read(PARTY_SPECIES[index]) for index in range(party_count)),
            hp_fraction=self.read_hp_fraction(party_count),
            in_battle=self.read_symbol("wIsInBattle") != 0,
            event_count=sum(event_bits),
            event_bits=event_bits,
            max_opponent_level=max((self.read(address) for address in OPPONENT_LEVELS), default=0),
        )

    def read(self, address: int) -> int:
        return int(self.pyboy.memory[address])

    def read_symbol(self, symbol: str) -> int:
        return self.read(self.resolve(symbol))

    def resolve(self, symbol: str) -> Any:
        if symbol in self._cache:
            return self._cache[symbol]
        try:
            address = self.pyboy.symbol_lookup(symbol)
        except Exception:
            address = RAM_SYMBOLS[symbol]
        self._cache[symbol] = address
        return address

    def read_event_bits(self) -> tuple[int, ...]:
        return tuple(
            int(bit)
            for address in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            for bit in f"{self.read(address):08b}"
        )

    def read_hp_fraction(self, party_count: int) -> float:
        if party_count <= 0:
            return 0.0
        hp_sum = sum(self.read_u16(address) for address in PARTY_HP[:party_count])
        max_hp_sum = sum(self.read_u16(address) for address in PARTY_MAX_HP[:party_count])
        return hp_sum / max(max_hp_sum, 1)

    def read_u16(self, address: int) -> int:
        return 256 * self.read(address) + self.read(address + 1)
