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

def action_name(action: int, actions: tuple[str, ...] = ACTIONS) -> str:
    return actions[int(action)]


class StepHandler:
    def __init__(
        self,
        pyboy: Any,
        action_frames: int = 12,
        release_frames: int = 6,
        wait_frames: int | None = None,
        render: bool = False,
        actions: tuple[str, ...] = ACTIONS,
    ) -> None:
        self.pyboy = pyboy
        self.action_frames = action_frames
        self.release_frames = release_frames if wait_frames is None else wait_frames
        self.render = render
        self.actions = actions

    def run(self, action: int) -> bool:
        button = action_name(action, self.actions)

        if button in {"noop", "wait"}:
            return self._tick_many(self.action_frames)

        self.pyboy.button_press(button)

        if not self._tick_many(self.action_frames):
            return False

        self.pyboy.button_release(button)
        return self._tick_many(self.release_frames)

    def wait(self, frames: int) -> bool:
        if frames <= 0:
            return True
        return self._tick_many(frames)

    def _tick_many(self, frames: int) -> bool:
        for _ in range(frames):
            if not bool(self.pyboy.tick(1, self.render, False)):
                return False
        return True
