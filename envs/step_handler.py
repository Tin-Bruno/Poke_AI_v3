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
    def __init__(
        self,
        pyboy: Any,
        action_frames: int = 12,
        wait_frames: int = 0,
        render: bool = False,
    ) -> None:
        self.pyboy = pyboy
        self.action_frames = action_frames
        self.wait_frames = wait_frames
        self.render = render

    def run(self, action: int) -> bool:
        button = action_name(action)
        if button == "noop":
            running = self.tick(self.action_frames)
        else:
            running = self.press_and_release(button)
        if self.wait_frames > 0:
            running = self.tick(self.wait_frames) and running
        return running

    def wait(self, frames: int) -> bool:
        if frames <= 0:
            return True
        return self.tick(frames)

    def press_and_release(self, button: str) -> bool:
        self.pyboy.button_press(button)
        running = self.tick(self.action_frames)
        self.pyboy.button_release(button)
        return self.tick(1) and running

    def tick(self, frames: int) -> bool:
        running = True
        for _ in range(max(0, frames)):
            if self.render:
                running = bool(self.pyboy.tick()) and running
            else:
                running = bool(self.pyboy.tick(1, False, False)) and running
        return running
