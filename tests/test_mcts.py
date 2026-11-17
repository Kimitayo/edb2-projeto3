"""Testes do MCTS clássico (algorithms/mcts.py).

Responsável: Gustavo (Integrante 3).
Execute a partir da raiz do projeto: python -m pytest tests/test_mcts.py -v
"""
import random

import chess
import pytest

from algorithms.base import Budget, Decision
from algorithms.mcts import (BudgetExhausted, Clock, MCTSAgent, PLAYOUT_LIMIT,
                             heuristic_white, playout)
from game import (initial_state, is_terminal, legal_actions,
                  reset_transition_count, transition_count)

# Posições de referência (FEN validadas)
MATE_IN_ONE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"  # 1. Ra8#
CHECKMATED_FEN = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"     # pretas já em mate
DRAW_FEN = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"            # K x K (material insuficiente)


# --- Decisão básica -----------------------------------------------------------
def test_returns_decision_with_legal_action():
    state = initial_state()
    decision = MCTSAgent(seed=1).choose_action(state, Budget(max_transitions=300))
    assert isinstance(decision, Decision)
    assert decision.action in legal_actions(state)


def test_tiny_budget_still_returns_legal_action():
    state = initial_state()
    decision = MCTSAgent(seed=1).choose_action(state, Budget(max_transitions=1))
    assert decision.action in legal_actions(state)


def test_mate_in_one_is_found():
    state = initial_state(MATE_IN_ONE_FEN)
    decision = MCTSAgent(seed=0).choose_action(state, Budget(max_transitions=5000))
    assert decision.action == chess.Move.from_uci("a1a8")


# --- Orçamento ----------------------------------------------------------------
def test_transitions_respect_budget():
    reset_transition_count()
    state = initial_state()
    decision = MCTSAgent(seed=2).choose_action(state, Budget(max_transitions=500))
    assert decision.extra["transitions"] <= 500
    # O contador global deve bater com o que o agente declarou
    assert transition_count() == decision.extra["transitions"]


def test_more_budget_means_more_iterations():
    state = initial_state()
    small = MCTSAgent(seed=3).choose_action(state, Budget(max_transitions=200))
    large = MCTSAgent(seed=3).choose_action(state, Budget(max_transitions=2000))
    assert large.extra["iterations"] > small.extra["iterations"]


def test_clock_without_limit_raises():
    with pytest.raises(ValueError):
        Clock(Budget())


def test_clock_with_zero_budget_raises_on_check():
    clock = Clock(Budget(max_transitions=0))
    with pytest.raises(BudgetExhausted):
        clock.check()


# --- Reprodutibilidade --------------------------------------------------------
def test_same_seed_gives_same_result():
    state = initial_state()
    budget = Budget(max_transitions=1000)
    d1 = MCTSAgent(seed=7).choose_action(state, budget)
    d2 = MCTSAgent(seed=7).choose_action(state, budget)
    assert d1.action == d2.action
    assert d1.extra["iterations"] == d2.extra["iterations"]


# --- Estados terminais --------------------------------------------------------
def test_terminal_root_raises():
    state = initial_state(CHECKMATED_FEN)
    assert is_terminal(state)
    with pytest.raises(ValueError):
        MCTSAgent(seed=0).choose_action(state, Budget(max_transitions=100))


def test_draw_position_is_terminal():
    assert is_terminal(initial_state(DRAW_FEN))


# --- Playout e heurística -----------------------------------------------------
def test_playout_stops_at_limit():
    state = initial_state()
    clock = Clock(Budget(max_transitions=10_000))
    value, steps = playout(state, random.Random(0), clock, limit=5)
    assert steps == 5
    assert -1.0 <= value <= 1.0


def test_playout_on_terminal_state_returns_utility():
    state = initial_state(CHECKMATED_FEN)  # pretas em mate: brancas vencem
    clock = Clock(Budget(max_transitions=10_000))
    value, steps = playout(state, random.Random(0), clock)
    assert steps == 0
    assert value == 1.0


def test_playout_respects_default_limit():
    state = initial_state()
    clock = Clock(Budget(max_transitions=10_000))
    _, steps = playout(state, random.Random(0), clock)
    assert steps <= PLAYOUT_LIMIT


def test_playout_with_zero_budget_raises():
    clock = Clock(Budget(max_transitions=0))
    with pytest.raises(BudgetExhausted):
        playout(initial_state(), random.Random(0), clock)


@pytest.mark.parametrize("fen", [None, MATE_IN_ONE_FEN, DRAW_FEN])
def test_heuristic_is_bounded(fen):
    value = heuristic_white(initial_state(fen))
    assert -1.0 <= value <= 1.0
