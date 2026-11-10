"""Regras da batalha — INTERFACE COMUM usada por todos os algoritmos.

Funções públicas (contrato do grupo):
    legal_actions(state)        -> tuple[int, ...]
    next_state(state, action)   -> BattleState (novo objeto)
    is_terminal(state)          -> bool
    get_winner(state)           -> 0 | 1 | None   (None = empate ou não terminal)
    utility(state, player)      -> +1.0 / 0.0 / -1.0   (só em estados terminais)
    evaluate(state, player)     -> float em [-1, 1]    (heurística, qualquer estado)

Ação = índice do ataque na lista `spec.moves` do Pokémon da vez.

Ordem de resolução de um ataque:
    1. gasta 1 PP (se limitado)
    2. aplica dano ao oponente (HP mínimo 0)
    3. aplica cura ao usuário (HP máximo = max_hp)
    4. ply += 1 e a vez passa ao oponente

Fim de jogo:
    - HP de um Pokémon chega a 0 -> o outro jogador vence
    - ply == max_plies sem nocaute -> empate
"""
from __future__ import annotations

from typing import Optional

from .state import BattleState, opponent

# --- Contador de transições (métrica comum "nós gerados") -------------------
# Toda chamada a next_state incrementa este contador. Os benchmarks usam
# reset_transition_count()/transition_count() para medir o custo de qualquer
# algoritmo da mesma forma, sem depender da contagem interna de cada um.
_transitions = 0


def reset_transition_count() -> None:
    global _transitions
    _transitions = 0


def transition_count() -> int:
    return _transitions


# --- Regras -----------------------------------------------------------------
def legal_actions(state: BattleState) -> tuple[int, ...]:
    if is_terminal(state):
        return ()
    mon = state.team[state.to_move]
    return tuple(i for i, pp in enumerate(mon.pp) if pp is None or pp > 0)


def next_state(state: BattleState, action: int) -> BattleState:
    global _transitions
    if action not in legal_actions(state):
        raise ValueError(f"Ação ilegal {action} no estado {state.key()}")
    _transitions += 1

    p, q = state.to_move, opponent(state.to_move)
    me, foe = state.team[p], state.team[q]
    move = me.spec.moves[action]

    pp = list(me.pp)
    if pp[action] is not None:
        pp[action] -= 1
    foe_hp = max(0, foe.hp - move.damage)
    my_hp = min(me.spec.max_hp, me.hp + move.heal)

    team = [None, None]
    team[p] = type(me)(me.spec, my_hp, tuple(pp))
    team[q] = type(foe)(foe.spec, foe_hp, foe.pp)
    return BattleState((team[0], team[1]), q, state.ply + 1, state.max_plies)


def is_terminal(state: BattleState) -> bool:
    return (state.team[0].fainted or state.team[1].fainted
            or state.ply >= state.max_plies)


def get_winner(state: BattleState) -> Optional[int]:
    if state.team[1].fainted:
        return 0
    if state.team[0].fainted:
        return 1
    return None  # empate (se terminal) ou jogo em andamento


def utility(state: BattleState, player: int) -> float:
    """Resultado exato de um estado TERMINAL do ponto de vista de `player`."""
    if not is_terminal(state):
        raise ValueError("utility() só é definida para estados terminais")
    w = get_winner(state)
    if w is None:
        return 0.0
    return 1.0 if w == player else -1.0


def evaluate(state: BattleState, player: int) -> float:
    """Heurística padrão em [-1, 1]: diferença das frações de HP.
    Em estados terminais devolve a utilidade exata. Cada algoritmo pode
    usar uma heurística própria, desde que documente."""
    if is_terminal(state):
        return utility(state, player)
    me, foe = state.team[player], state.team[opponent(player)]
    return me.hp / me.spec.max_hp - foe.hp / foe.spec.max_hp


# --- Características do estado (usadas pela busca híbrida) ------------------
def state_features(state: BattleState) -> dict:
    """Atributos numéricos do estado, do ponto de vista do jogador da vez.
    O Integrante 4 pode usar/estender para a política de seleção."""
    p = state.to_move
    me, foe = state.team[p], state.team[opponent(p)]
    return {
        "my_hp_frac": me.hp / me.spec.max_hp,
        "foe_hp_frac": foe.hp / foe.spec.max_hp,
        "hp_diff": me.hp / me.spec.max_hp - foe.hp / foe.spec.max_hp,
        "branching": len(legal_actions(state)),
        "foe_branching": sum(1 for pp in foe.pp if pp is None or pp > 0),
        "plies_left": state.max_plies - state.ply,
        "my_limited_pp": sum(pp for pp in me.pp if pp is not None),
        "foe_limited_pp": sum(pp for pp in foe.pp if pp is not None),
    }


# --- Utilidades para depuração/visualização ---------------------------------
def to_dict(state: BattleState) -> dict:
    return {
        "ply": state.ply, "max_plies": state.max_plies, "to_move": state.to_move,
        "team": [{"name": m.spec.name, "hp": m.hp, "max_hp": m.spec.max_hp,
                  "moves": [{"name": mv.name, "pp": pp}
                            for mv, pp in zip(m.spec.moves, m.pp)]}
                 for m in state.team],
    }


def render(state: BattleState) -> str:
    lines = [f"ply {state.ply}/{state.max_plies} | vez do jogador {state.to_move}"]
    for p, m in enumerate(state.team):
        pps = ", ".join(f"{mv.name}({'∞' if pp is None else pp})"
                        for mv, pp in zip(m.spec.moves, m.pp))
        lines.append(f"  J{p} {m.spec.name}: {m.hp}/{m.spec.max_hp} HP | {pps}")
    return "\n".join(lines)
