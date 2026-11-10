"""Executa uma batalha completa entre dois agentes e registra o histórico.

Demonstração: python -m algorithms.match [cenario]   (guloso x aleatório)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from game import (BattleState, is_terminal, next_state, get_winner,
                  reset_transition_count, transition_count)
from .base import Agent, Budget


@dataclass
class Step:
    ply: int
    player: int
    action: int
    move_name: str
    transitions: int   # chamadas a next_state feitas pelo agente nesta decisão
    time_s: float
    nodes_expanded: int
    extra: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    winner: Optional[int]      # 0, 1 ou None (empate)
    plies: int
    steps: list[Step]
    final_state: BattleState


def play_match(agents: tuple[Agent, Agent], state: BattleState,
               budgets: tuple[Budget, Budget] = (Budget(), Budget())) -> MatchResult:
    steps: list[Step] = []
    while not is_terminal(state):
        p = state.to_move
        reset_transition_count()
        t0 = time.perf_counter()
        d = agents[p].choose_action(state, budgets[p])
        dt = time.perf_counter() - t0
        used = transition_count()
        name = state.team[p].spec.moves[d.action].name
        state = next_state(state, d.action)
        steps.append(Step(state.ply - 1, p, d.action, name, used, dt,
                          d.nodes_expanded, d.extra))
    return MatchResult(get_winner(state), state.ply, steps, state)


if __name__ == "__main__":  # demonstração rápida
    import sys
    from game import render
    from game.scenarios import SCENARIOS
    from .base import GreedyAgent, RandomAgent

    name = sys.argv[1] if len(sys.argv) > 1 else "trap_save_heal"
    state = SCENARIOS[name]()
    print(f"Cenário: {name}"); print(render(state)); print()
    r = play_match((GreedyAgent(), RandomAgent(seed=42)), state)
    width = max(len(st.move_name) for st in r.steps)
    s = state
    for st in r.steps:  # reaplica as ações para mostrar o HP após cada jogada
        s = next_state(s, st.action)
        hp = " | ".join(f"{m.spec.name} {m.hp}/{m.spec.max_hp}" for m in s.team)
        print(f"ply {st.ply:2}: J{st.player} {state.team[st.player].spec.name} "
              f"usa {st.move_name:<{width}}  -> {hp}")
    print(); print(render(r.final_state))
    print("Resultado:", "empate" if r.winner is None else f"vitória do J{r.winner}")
