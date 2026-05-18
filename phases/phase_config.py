"""Configuracao declarativa das fases de treino."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseConfig:
    id: str
    name: str
    state: str
    model: str
    max_steps: int
    rewards: tuple[str, ...]
    success: str
    target_map: int | None = None
    target_position_map: int | None = None
    target_x: int | None = None
    target_y: int | None = None
    target_radius: int = 0
    target_badges: int | None = None
    blocked_move_penalty: float = 0.0
    actions: tuple[str, ...] | None = None
    waypoints: tuple[tuple[int, int, int], ...] = ()


PHASES: dict[str, PhaseConfig] = {
    "phase1": PhaseConfig(
        id="phase1",
        name="Sair do quarto",
        state="phase1_start.state",
        model="models/phase1_room_exit.zip",
        max_steps=400,
        rewards=("target_position", "target_map", "new_map"),
        success="target_map",
        target_map=37,
        target_position_map=38,
        target_x=7,
        target_y=1,
        blocked_move_penalty=-0.03,
        actions=("up", "down", "left", "right", "noop"),
    ),
    "phase2": PhaseConfig(
        id="phase2",
        name="Sair da casa",
        state="phase2_start.state",
        model="models/phase2_house_exit.zip",
        max_steps=500,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=0,
        blocked_move_penalty=-0.03,
        actions=("up", "down", "left", "right", "noop"),
    ),
    "phase3": PhaseConfig(
        id="phase3",
        name="Ativar evento do Professor Oak",
        state="phase3_start.state",
        model="models/phase3_oak_event.zip",
        max_steps=600,
        rewards=("waypoint", "dialog"),
        success="event_count_increase",
        target_map=0,
        target_x=10,
        target_y=1,
        blocked_move_penalty=-0.03,
        actions=("up", "down", "left", "right", "noop"),
        waypoints=((0, 8, 2), (0, 10, 2), (0, 10, 1)),
    ),
    "phase4": PhaseConfig(
        id="phase4",
        name="Ser levado ao laboratorio",
        state="phase4_start.state",
        model="models/phase4_to_lab.zip",
        max_steps=800,
        rewards=("target_position", "dialog"),
        success="target_position",
        target_map=40,
        target_x=5,
        target_y=11,
        blocked_move_penalty=-0.03,
        actions=("a", "b", "noop"),
    ),
    "phase5": PhaseConfig(
        id="phase5",
        name="Passar dialogo e escolher starter",
        state="phase5_start.state",
        model="models/phase5_choose_starter.zip",
        max_steps=1800,
        rewards=("dialog", "waypoint"),
        success="party_count_increase",
        target_map=40,
        blocked_move_penalty=-0.03,
        actions=("up", "down", "left", "right", "a", "noop"),
        waypoints=(
            (40, 5, 10),
            (40, 5, 8),
            (40, 5, 6),
            (40, 5, 4),
            (40, 6, 4),
            (40, 7, 4),
            (40, 8, 4),
        ),
    ),
    "phase6": PhaseConfig(
        id="phase6",
        name="Passar dialogo do rival",
        state="phase6_start.state",
        model="models/phase6_rival_dialog.zip",
        max_steps=800,
        rewards=("dialog",),
        success="dialog_or_map_change",
    ),
    "phase7": PhaseConfig(
        id="phase7",
        name="Vencer primeira batalha",
        state="phase7_start.state",
        model="models/phase7_first_battle.zip",
        max_steps=2000,
        rewards=("battle", "dialog"),
        success="dialog_or_map_change",
    ),
    "phase8": PhaseConfig(
        id="phase8",
        name="Sair do laboratorio",
        state="phase8_start.state",
        model="models/phase8_exit_lab.zip",
        max_steps=800,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=0,
    ),
    "phase9": PhaseConfig(
        id="phase9",
        name="Ir para Rota 1",
        state="phase9_start.state",
        model="models/phase9_route_1.zip",
        max_steps=1200,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=12,
    ),
}


def get_phase(phase_id: str) -> PhaseConfig:
    try:
        return PHASES[phase_id]
    except KeyError as exc:
        valid = ", ".join(PHASES)
        raise ValueError(f"Fase desconhecida: {phase_id}. Fases validas: {valid}") from exc
