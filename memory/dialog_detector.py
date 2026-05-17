"""Detectores simples para saber se ha caixa de dialogo na tela."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DialogDetection:
    visual: bool
    score: float
    bottom_white: float
    bottom_dark: float
    top_edge_dark: float
    bottom_edge_dark: float
    left_edge_dark: float
    right_edge_dark: float
    window_x: int | None = None
    window_y: int | None = None
    window_on_screen: bool | None = None
    bottom_unique_tiles: int | None = None

    def to_info(self) -> dict[str, bool | float | int | None]:
        return {
            "dialog_visual": self.visual,
            "dialog_score": self.score,
            "bottom_white": self.bottom_white,
            "bottom_dark": self.bottom_dark,
            "top_edge_dark": self.top_edge_dark,
            "bottom_edge_dark": self.bottom_edge_dark,
            "left_edge_dark": self.left_edge_dark,
            "right_edge_dark": self.right_edge_dark,
            "window_x": self.window_x,
            "window_y": self.window_y,
            "window_on_screen": self.window_on_screen,
            "bottom_unique_tiles": self.bottom_unique_tiles,
        }


def detect_dialog_from_pyboy(pyboy: Any) -> DialogDetection:
    screen = np.asarray(pyboy.screen.ndarray[:, :, :3])
    detection = detect_dialog_from_screen(screen)
    return enrich_with_pyboy_state(detection, pyboy)


def detect_dialog_from_screen(screen: np.ndarray) -> DialogDetection:
    """Detecta a caixa de texto padrao na parte inferior da tela.

    Pokemon Red desenha caixas de dialogo como uma janela branca com borda escura.
    A heuristica olha apenas para a regiao inferior da tela e mede se ha linhas
    horizontais/verticais escuras onde a borda normalmente aparece.
    """

    if screen.ndim == 3:
        gray = screen[:, :, :3].mean(axis=2)
    else:
        gray = screen

    height, width = gray.shape
    dark = gray < 80
    white = gray > 175

    # Caixa de texto padrao ocupa aproximadamente os ultimos 48 pixels.
    top = max(0, height - 48)
    bottom = height - 1
    left = 0
    right = width - 1
    region = gray[top:height, :]
    region_dark = dark[top:height, :]
    region_white = white[top:height, :]

    top_edge_dark = float(dark[top : min(top + 5, height), :].mean())
    bottom_edge_dark = float(dark[max(0, bottom - 4) : bottom + 1, :].mean())
    left_edge_dark = float(dark[top:height, left : min(left + 5, width)].mean())
    right_edge_dark = float(dark[top:height, max(0, right - 4) : right + 1].mean())
    bottom_white = float(region_white.mean())
    bottom_dark = float(region_dark.mean())

    # Interior sem bordas. Caixa de texto tem fundo claro; tela preta embaixo nao.
    inner = region[8:-8, 8:-8] if region.shape[0] > 16 and region.shape[1] > 16 else region
    inner_white = float((inner > 175).mean())

    # Borda escura + interior claro e o sinal mais forte de textbox.
    edge_score = (top_edge_dark + bottom_edge_dark + left_edge_dark + right_edge_dark) / 4
    score = (edge_score * 0.45) + (inner_white * 0.55)
    visual = score >= 0.35 and inner_white >= 0.20 and top_edge_dark >= 0.15

    return DialogDetection(
        visual=visual,
        score=score,
        bottom_white=bottom_white,
        bottom_dark=bottom_dark,
        top_edge_dark=top_edge_dark,
        bottom_edge_dark=bottom_edge_dark,
        left_edge_dark=left_edge_dark,
        right_edge_dark=right_edge_dark,
    )


def enrich_with_pyboy_state(detection: DialogDetection, pyboy: Any) -> DialogDetection:
    window_x = None
    window_y = None
    window_on_screen = None
    bottom_unique_tiles = None

    try:
        (_, _), (window_x, window_y) = pyboy.screen.get_tilemap_position()
        window_on_screen = -7 <= int(window_x) < 160 and 0 <= int(window_y) < 144
    except Exception:
        pass

    try:
        bottom_tiles = np.asarray(pyboy.tilemap_background[0:20, 12:18], dtype=np.uint32)
        bottom_unique_tiles = int(np.unique(bottom_tiles).size)
    except Exception:
        pass

    return DialogDetection(
        visual=detection.visual,
        score=detection.score,
        bottom_white=detection.bottom_white,
        bottom_dark=detection.bottom_dark,
        top_edge_dark=detection.top_edge_dark,
        bottom_edge_dark=detection.bottom_edge_dark,
        left_edge_dark=detection.left_edge_dark,
        right_edge_dark=detection.right_edge_dark,
        window_x=window_x,
        window_y=window_y,
        window_on_screen=window_on_screen,
        bottom_unique_tiles=bottom_unique_tiles,
    )
