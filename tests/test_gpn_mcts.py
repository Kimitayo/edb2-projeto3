"""Testes do GPN-MCTS (algorithms/gpn_mcts.py)."""
import chess
import pytest

from algorithms.base import Budget
from algorithms.gpn_mcts import (FORMULAS, INF, GNode, GPNMCTSAgent, _bias,
                                 _backup_pn, _leaf_pn, _root_solved,
                                 _update_pn)
from game import initial_state, legal_actions, next_state

MATE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
DRAW_FEN = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
START_FEN = chess.STARTING_FEN


# --- Agente: contratos básicos ------------------------------------------------
def test_returns_legal_action():
    state = initial_state(START_FEN)
    d = GPNMCTSAgent(seed=0).choose_action(state, Budget(max_transitions=2000))
    assert d.action.uci() in {a.uci() for a in legal_actions(state)}


def test_tiny_budget_still_returns_legal_action():
    state = initial_state(START_FEN)
    d = GPNMCTSAgent(seed=0).choose_action(state, Budget(max_transitions=1))
    assert d.action.uci() in {a.uci() for a in legal_actions(state)}


def test_terminal_root_raises():
    # Afogamento: pretas sem lances e sem xeque (estado terminal)
    state = initial_state("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    with pytest.raises(ValueError):
        GPNMCTSAgent(seed=0).choose_action(state, Budget(max_transitions=100))


def test_invalid_formula_raises():
    with pytest.raises(ValueError, match="Fórmula inválida"):
        GPNMCTSAgent(formula="xyz")


@pytest.mark.parametrize("formula", FORMULAS)
def test_all_formulas_find_mate_in_one(formula):
    d = GPNMCTSAgent(seed=0, formula=formula).choose_action(
        initial_state(MATE_FEN), Budget(max_transitions=5000))
    assert d.action.uci() == "a1a8"
    assert d.extra["formula"] == formula


def test_transitions_respect_budget():
    budget = 3000
    d = GPNMCTSAgent(seed=0).choose_action(initial_state(START_FEN),
                                           Budget(max_transitions=budget))
    assert d.extra["transitions"] <= budget


def test_more_budget_means_more_iterations():
    small = GPNMCTSAgent(seed=0).choose_action(initial_state(START_FEN),
                                               Budget(max_transitions=500))
    large = GPNMCTSAgent(seed=0).choose_action(initial_state(START_FEN),
                                               Budget(max_transitions=5000))
    assert large.extra["iterations"] > small.extra["iterations"]


def test_same_seed_gives_same_result():
    state = initial_state(START_FEN)
    a = GPNMCTSAgent(seed=7).choose_action(state, Budget(max_transitions=2000))
    b = GPNMCTSAgent(seed=7).choose_action(state, Budget(max_transitions=2000))
    assert a.action.uci() == b.action.uci()
    assert a.extra["iterations"] == b.extra["iterations"]
    assert a.extra["playouts"] == b.extra["playouts"]


# --- Valores de folha e prova -------------------------------------------------
def test_leaf_pn_for_checkmate_favors_winner():
    board = chess.Board(MATE_FEN)
    board.push_uci("a1a8")  # mate: pretas a mover e sem lances
    pn = _leaf_pn(initial_state(board.fen()))
    assert pn[chess.WHITE] == 0.0
    assert pn[chess.BLACK] == INF


def test_leaf_pn_for_draw_is_inf_for_both():
    pn = _leaf_pn(initial_state(DRAW_FEN))
    assert pn[chess.WHITE] == INF
    assert pn[chess.BLACK] == INF


def test_leaf_pn_for_nonterminal_is_one_for_both():
    pn = _leaf_pn(initial_state(START_FEN))
    assert pn[chess.WHITE] == 1.0
    assert pn[chess.BLACK] == 1.0


def test_root_solved_when_mover_has_proof():
    root = GNode(initial_state(START_FEN))
    root.pn[chess.WHITE] = 0.0  # brancas (a mover) provaram vitória
    assert _root_solved(root)


def test_root_not_solved_without_proof():
    root = GNode(initial_state(START_FEN))
    assert not _root_solved(root)


# --- Propagação de pn ---------------------------------------------------------
def _expanded_root(fen: str) -> GNode:
    """Raiz com todos os filhos criados e nenhum lance por tentar."""
    state = initial_state(fen)
    root = GNode(state)
    for a in legal_actions(state):
        root.children[a] = GNode(next_state(state, a), root, a)
    root.untried = []
    return root


def test_or_node_takes_minimum_of_children():
    root = _expanded_root(START_FEN)  # brancas a mover: nó OR
    first = next(iter(root.children.values()))
    first.pn[chess.WHITE] = 0.0
    assert _update_pn(root, chess.WHITE) is True
    assert root.pn[chess.WHITE] == 0.0


def test_and_node_sums_children_for_opponent():
    root = _expanded_root(START_FEN)  # brancas a mover: pretas são AND aqui
    k = len(root.children)
    assert _update_pn(root, chess.BLACK) is True
    assert root.pn[chess.BLACK] == float(k)  # cada filho vale 1


def test_untried_children_count_as_one():
    root = _expanded_root(START_FEN)
    root.untried = [next(iter(root.children))]  # um lance ainda não tentado
    root.children.pop(root.untried[0])
    _update_pn(root, chess.BLACK)
    assert root.pn[chess.BLACK] == float(len(root.children) + 1)


def test_backup_propagates_to_ancestors():
    root = _expanded_root(MATE_FEN)  # brancas a mover, há mate em um
    mate_action = chess.Move.from_uci("a1a8")
    mate_child = next(ch for a, ch in root.children.items() if a.uci() == "a1a8")
    mate_child.pn = {chess.WHITE: 0.0, chess.BLACK: INF}
    _backup_pn([root, mate_child])
    assert root.pn[chess.WHITE] == 0.0


# --- Viés UCT-PN --------------------------------------------------------------
@pytest.mark.parametrize("formula", FORMULAS)
def test_bias_is_bounded(formula):
    bias = _bias([0.0, 1.0, 3.0, 7.0], formula)
    assert all(0.0 <= b <= 1.0 for b in bias)


@pytest.mark.parametrize("formula", FORMULAS)
def test_bias_is_zero_for_inf(formula):
    bias = _bias([2.0, INF, 5.0], formula)
    assert bias[1] == 0.0


@pytest.mark.parametrize("formula", FORMULAS)
def test_bias_all_inf_returns_zeros(formula):
    assert _bias([INF, INF], formula) == [0.0, 0.0]


def test_pnmax_prefers_smaller_pn():
    bias = _bias([1.0, 4.0], "max")
    assert bias[0] > bias[1]


def test_pnrank_prefers_smaller_pn():
    bias = _bias([1.0, 4.0, 9.0], "rank")
    assert bias[0] > bias[1] > bias[2]
