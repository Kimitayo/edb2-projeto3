"""Gabarito dos cenários (calculado pelo solver exato tests/oracle.py).

Os outros integrantes podem usar EXPECTED para validar seus algoritmos:
    from tests.test_scenarios import EXPECTED
"""
import sys

import pytest

from algorithms.base import GreedyAgent, RandomAgent, Budget
from algorithms.match import play_match
from game.scenarios import SCENARIOS
from tests.oracle import solve, optimal_actions

sys.setrecursionlimit(10_000)

# nome: (valor para o jogador da vez na raiz, jogadas ótimas na raiz)
EXPECTED = {
    "sanity_mvp": (+1, {1}),
    "trap_save_heal": (+1, {0}),
    "trap_save_nuke": (+1, {0}),
    "trap_2v3": (+1, {0}),
    "trap_3v3": (+1, {0}),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_oracle_answer_key(name):
    s = SCENARIOS[name]()
    value, best = EXPECTED[name]
    assert solve(s) == value
    assert optimal_actions(s) == best


@pytest.mark.parametrize("name", [n for n in EXPECTED if n.startswith("trap")])
def test_greedy_falls_into_traps(name):
    s = SCENARIOS[name]()
    assert GreedyAgent().choose_action(s, Budget()).action not in EXPECTED[name][1]


@pytest.mark.parametrize("name", SCENARIOS)
def test_matches_terminate(name):
    r = play_match((RandomAgent(1), RandomAgent(2)), SCENARIOS[name]())
    assert r.plies <= r.final_state.max_plies
    assert r.winner in (0, 1, None)
