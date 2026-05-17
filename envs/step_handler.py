"""Acoes discretas e execucao de botoes no PyBoy."""

from __future__ import annotations

from typing import Any


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
    return ACTIONS[int(action)]


class StepHandler:
    def __init__(self, pyboy: Any, action_frames: int = 12, render: bool = False) -> None:
        self.pyboy = pyboy
        self.action_frames = action_frames
        self.render = render

    def run(self, action: int) -> bool:
        button = action_name(action)
        if button != "noop":
            self.pyboy.button(button, self.action_frames)
        return bool(self.pyboy.tick(self.action_frames, self.render, False))
