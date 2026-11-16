"""Executa uma partida completa entre dois agentes e registra o histórico.

Demonstração: python -m algorithms.match [seed]   (aleatório x aleatório)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import chess

from game import (Action, Outcome, get_outcome, initial_state, is_terminal,
                  next_state, reset_transition_count, transition_count)
from .base import Agent, Budget


@dataclass
class Step:
    index: int         # número da decisão na partida (0, 1, ...)
    player: bool       # chess.WHITE ou chess.BLACK
    action: Action
    transitions: int   # chamadas a next_state feitas pelo agente nesta decisão
    time_s: float
    nodes_expanded: int
    extra: dict = field(default_factory=dict)


@dataclass
class MatchResult:
    outcome: Outcome
    plies: int                  # meios-lances jogados
    steps: list[Step]
    final_state: chess.Board


def play_match(agents: tuple[Agent, Agent], state: chess.Board | None = None,
               budgets: tuple[Budget, Budget] = (Budget(), Budget())) -> MatchResult:
    """agents[0] joga de brancas e agents[1] de pretas (idem para budgets)."""
    state = initial_state() if state is None else state
    steps: list[Step] = []
    while not is_terminal(state):
        p = state.turn
        i = 0 if p == chess.WHITE else 1
        reset_transition_count()
        t0 = time.perf_counter()
        d = agents[i].choose_action(state, budgets[i])
        dt = time.perf_counter() - t0
        used = transition_count()
        state = next_state(state, d.action)
        steps.append(Step(len(steps), p, d.action, used, dt, d.nodes_expanded, d.extra))
    return MatchResult(get_outcome(state), len(state.move_stack), steps, state)


if __name__ == "__main__":  # demonstração rápida
    import sys
    from game import render
    from .base import RandomAgent

    sys.stdout.reconfigure(encoding="utf-8")  # peças Unicode em qualquer terminal
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    r = play_match((RandomAgent(seed), RandomAgent(seed + 1)))
    print(render(r.final_state))
    print(f"Resultado: {r.outcome.result.value} ({r.outcome.reason}) | "
          f"{r.plies} meios-lances | {len(r.steps)} decisões")
