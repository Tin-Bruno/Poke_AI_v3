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
    target_event_count_delta: int = 1
    target_badges: int | None = None
    blocked_move_penalty: float = 0.0
    actions: tuple[str, ...] | None = None
    waypoints: tuple[tuple[int, int, int], ...] = ()
    save_actions: tuple[str, ...] = ()
    scripted_actions: tuple[str, ...] = ()


MOVE_ACTIONS = ("up", "down", "left", "right", "noop")
DIALOG_ACTIONS = ("b", "noop")
DIALOG_CONFIRM_ACTIONS = ("a", "b", "noop")
DIALOG_RELEASE_ACTIONS = ("b", "down", "noop")


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
        actions=MOVE_ACTIONS,
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
        actions=MOVE_ACTIONS,
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
        actions=MOVE_ACTIONS,
        waypoints=((0, 8, 2), (0, 10, 2), (0, 10, 1)),
    ),
    "phase4": PhaseConfig(
        id="phase4",
        name="Ser levado ao laboratorio",
        state="phase4_start.state",
        model="models/phase4_to_lab.zip",
        max_steps=800,
        rewards=("target_map", "dialog"),
        success="target_map",
        target_map=40,
        blocked_move_penalty=-0.03,
        actions=DIALOG_CONFIRM_ACTIONS,
    ),
    "phase5": PhaseConfig(
        id="phase5",
        name="Passar dialogo inicial do laboratorio",
        state="phase5_start.state",
        model="models/phase5_lab_dialog.zip",
        max_steps=500,
        rewards=("dialog", "target_position"),
        success="target_position_after_event_count",
        target_map=40,
        target_x=5,
        target_y=3,
        target_event_count_delta=3,
        blocked_move_penalty=-0.03,
        actions=DIALOG_RELEASE_ACTIONS,
        save_actions=("b",),
    ),
    "phase5b": PhaseConfig(
        id="phase5b",
        name="Ir ate a frente da Pokebola",
        state="phase5b_start.state",
        model="models/phase5b_to_pokeball.zip",
        max_steps=1200,
        rewards=("waypoint", "target_position"),
        success="target_position",
        target_map=40,
        target_x=6,
        target_y=4,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
        save_actions=("up",),
    ),
    "phase5c": PhaseConfig(
        id="phase5c",
        name="Escolher starter",
        state="phase5c_start.state",
        model="models/phase5c_choose_starter.zip",
        max_steps=800,
        rewards=("dialog", "party"),
        success="party_count_increase",
        target_map=40,
        actions=DIALOG_CONFIRM_ACTIONS,
    ),
    "phase6": PhaseConfig(
        id="phase6",
        name="Passar dialogo do rival",
        state="phase6_start.state",
        model="models/phase6_rival_dialog.zip",
        max_steps=800,
        rewards=("dialog",),
        success="dialog_or_map_change",
        actions=DIALOG_ACTIONS,
    ),
    "phase7": PhaseConfig(
        id="phase7",
        name="Iniciar batalha do rival",
        state="phase7_start.state",
        model="models/phase7_start_rival_battle.zip",
        max_steps=1200,
        rewards=("waypoint", "dialog"),
        success="battle_started",
        target_map=40,
        target_x=5,
        target_y=6,
        blocked_move_penalty=-0.03,
        actions=("up", "down", "left", "right", "b", "noop"),
        waypoints=(
            (40, 6, 5),
            (40, 5, 5),
            (40, 5, 6),
        ),
    ),
    "phase7b": PhaseConfig(
        id="phase7b",
        name="Vencer primeira batalha",
        state="phase7b_start.state",
        model="models/phase7b_first_battle.zip",
        max_steps=2500,
        rewards=("battle", "dialog"),
        success="battle_won",
        actions=DIALOG_CONFIRM_ACTIONS,
    ),
    "phase8": PhaseConfig(
        id="phase8",
        name="Passar dialogo pos-batalha",
        state="phase8_start.state",
        model="models/phase8_after_battle_dialog.zip",
        max_steps=800,
        rewards=("target_position",),
        success="target_position",
        target_map=40,
        target_x=5,
        target_y=7,
        blocked_move_penalty=-0.03,
        actions=DIALOG_RELEASE_ACTIONS,
        scripted_actions=("b",) * 20 + ("down",) * 4,
    ),
    "phase8b": PhaseConfig(
        id="phase8b",
        name="Sair do laboratorio",
        state="phase8b_start.state",
        model="models/phase8b_exit_lab.zip",
        max_steps=800,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_position_map=38,
        target_x=7,
        target_y=1,
        target_map=0,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
    ),
    "phase9": PhaseConfig(
        id="phase9",
        name="Ir para Rota 1",
        state="phase9_start.state",
        model="models/phase9_route_1.zip",
        max_steps=1200,
        rewards=("waypoint", "target_map"),
        success="target_map",
        target_map=12,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
        waypoints=(
            (0, 9, 12),
            (0, 9, 8),
            (0, 9, 2),
            (0, 10, 1),
        ),
    ),
    "phase10": PhaseConfig(
        id="phase10",
        name="Rota 1 ate Viridian",
        state="phase10_start.state",
        model="models/phase10_viridian.zip",
        max_steps=1800,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=1,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
    ),
    "phase11": PhaseConfig(
        id="phase11",
        name="Viridian ate Rota 2",
        state="phase11_start.state",
        model="models/phase11_route_2.zip",
        max_steps=2500,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=13,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
    ),
    "phase12": PhaseConfig(
        id="phase12",
        name="Rota 2 ate Viridian Forest",
        state="phase12_start.state",
        model="models/phase12_viridian_forest.zip",
        max_steps=2500,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=51,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
    ),
    "phase13": PhaseConfig(
        id="phase13",
        name="Atravessar Viridian Forest",
        state="phase13_start.state",
        model="models/phase13_forest_exit.zip",
        max_steps=6000,
        rewards=("target_map", "new_map", "battle"),
        success="target_map",
        target_map=13,
        blocked_move_penalty=-0.03,
        actions=("noop", "up", "down", "left", "right", "a", "b"),
    ),
    "phase14": PhaseConfig(
        id="phase14",
        name="Rota 2 ate Pewter",
        state="phase14_start.state",
        model="models/phase14_pewter.zip",
        max_steps=2500,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=2,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
    ),
    "phase14b": PhaseConfig(
        id="phase14b",
        name="Pewter ate ginasio",
        state="phase14b_start.state",
        model="models/phase14b_pewter_gym.zip",
        max_steps=2500,
        rewards=("target_map", "new_map"),
        success="target_map",
        target_map=54,
        blocked_move_penalty=-0.03,
        actions=MOVE_ACTIONS,
    ),
    "phase15": PhaseConfig(
        id="phase15",
        name="Vencer Brock",
        state="phase15_start.state",
        model="models/phase15_brock.zip",
        max_steps=7000,
        rewards=("battle", "dialog"),
        success="badge_count",
        target_badges=1,
        actions=DIALOG_CONFIRM_ACTIONS,
    ),
    "freeplay_route1": PhaseConfig(
        id="freeplay_route1",
        name="Freeplay depois da Rota 1",
        state="freeplay_start.state",
        model="models/freeplay_route1.zip",
        max_steps=5000,
        rewards=("new_map", "dialog", "battle", "party"),
        success="never",
        blocked_move_penalty=-0.02,
        actions=("noop", "up", "down", "left", "right", "a", "b"),
    ),
}


def get_phase(phase_id: str) -> PhaseConfig:
    try:
        return PHASES[phase_id]
    except KeyError as exc:
        valid = ", ".join(PHASES)
        raise ValueError(f"Fase desconhecida: {phase_id}. Fases validas: {valid}") from exc
