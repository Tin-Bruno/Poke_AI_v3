"""Mapeamento discreto de acoes para botoes do Game Boy."""

from __future__ import annotations

ACTIONS: tuple[str, ...] = (
    "noop",
    "up",
    "down",
    "left",
    "right",
    "a",
    "b",
    "start",
    "select",
)


def action_name(action: int) -> str:
    """Retorna o nome da acao discreta."""
    return ACTIONS[int(action)]
