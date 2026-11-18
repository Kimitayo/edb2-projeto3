"""Testes do GPN-MCTS (algorithms/gpn_mcts.py)."""
import math

import chess
import pytest

from algorithms.base import Budget
from algorithms.gpn_mcts import (DEFAULT_CPN, FORMULAS, INF, GNode, GPNMCTSAgent,
                                 GStats, _backup_pn, _bias, _bounds_solved,
                                 _dominated, _leaf_pn, _root_solved,
                                 _sync_pn_with_bounds, _update_bounds,
                                 _update_pn)
from game import initial_state, legal_actions, next_state

MATE_FEN = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
DRAW_FEN = "8/8/8/4k3/8/8/8/4K3 w - - 0 1"
BLACK_MATE_FEN = "r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"  # 1... Ra1#
START_FEN = chess.STARTING_FEN


# --- Configuração do planejamento (PNRank, C = sqrt(2), Cpn = 1.0, SB) ---------
def test_default_config_matches_project_plan():
    agent = GPNMCTSAgent(seed=0)  # o runner instancia só com a seed
    assert agent.formula == "rank"
    assert agent.c == pytest.approx(math.sqrt(2))
    assert agent.cpn == DEFAULT_CPN == 1.0
    assert agent.score_bounded is True


def test_decision_reports_configuration():
    d = GPNMCTSAgent(seed=0).choose_action(
        initial_state(START_FEN),
        Budget(max_transitions=3000),
    )
    assert d.extra["formula"] == "rank"
    assert d.extra["score_bounded"] is True
    assert d.extra["cpn"] == 1.0
    assert d.extra["c"] == pytest.approx(math.sqrt(2))
    assert d.extra["proof_updates"] > 0
    assert d.extra["pnrank_sorts"] > 0


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
    assert d.transitions <= budget


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


# --- Score-Bounded ------------------------------------------------------------
def test_terminal_node_has_exact_bounds():
    board = chess.Board(MATE_FEN)
    board.push_uci("a1a8")
    node = GNode(initial_state(board.fen()))
    assert node.pess == node.opti == 1.0


def test_nonterminal_node_starts_with_full_interval():
    node = GNode(initial_state(START_FEN))
    assert (node.pess, node.opti) == (-1.0, 1.0)


def test_max_node_bounds_follow_best_child():
    root = _expanded_root(MATE_FEN)  # brancas: MAX, há mate em um
    assert _update_bounds(root) is True
    assert root.pess == 1.0 and root.opti == 1.0
    assert _bounds_solved(root)


def test_min_node_bounds_follow_best_child():
    root = _expanded_root(BLACK_MATE_FEN)  # pretas: MIN, há mate em um
    assert _update_bounds(root) is True
    assert root.pess == -1.0 and root.opti == -1.0


def test_untried_moves_keep_bounds_open():
    root = _expanded_root(START_FEN)
    root.untried = [next(iter(root.children))]
    root.children.pop(root.untried[0])
    _update_bounds(root)
    assert root.opti == 1.0  # lance não tentado ainda pode vencer


def test_dominated_child_is_pruned_at_max_node():
    root = _expanded_root(START_FEN)  # brancas: MAX
    child = next(iter(root.children.values()))
    root.pess = 0.0
    child.opti = -0.5
    assert _dominated(root, child)
    child.opti = 0.0
    assert not _dominated(root, child)  # desigualdade estrita


def test_dominated_child_is_pruned_at_min_node():
    root = _expanded_root(BLACK_MATE_FEN)  # pretas: MIN
    child = next(iter(root.children.values()))
    root.opti = 0.0
    child.pess = 0.5
    assert _dominated(root, child)


def test_bounds_proving_draw_sets_pn_inf_for_both():
    node = GNode(initial_state(START_FEN))
    node.pess = node.opti = 0.0
    _sync_pn_with_bounds(node)
    assert node.pn[chess.WHITE] == INF and node.pn[chess.BLACK] == INF


def test_bounds_proving_white_win_sets_pn():
    node = GNode(initial_state(START_FEN))
    node.pess = node.opti = 1.0
    _sync_pn_with_bounds(node)
    assert node.pn[chess.WHITE] == 0.0 and node.pn[chess.BLACK] == INF


def test_backup_with_score_bounded_propagates_bounds():
    root = _expanded_root(MATE_FEN)
    mate_child = next(ch for a, ch in root.children.items() if a.uci() == "a1a8")
    stats = GStats()
    _backup_pn([root, mate_child], stats, score_bounded=True)
    assert root.pess == 1.0 and root.pn[chess.WHITE] == 0.0
    assert stats.proof_updates > 0


def test_root_solved_by_bounds_without_pn():
    root = GNode(initial_state(START_FEN))
    root.pess = root.opti = 0.0
    assert _root_solved(root)


def test_black_mate_in_one_is_found():
    d = GPNMCTSAgent(seed=0).choose_action(initial_state(BLACK_MATE_FEN),
                                           Budget(max_transitions=5000))
    assert d.action.uci() == "a8a1"


def test_score_bounded_can_be_disabled():
    d = GPNMCTSAgent(seed=0, score_bounded=False).choose_action(
        initial_state(START_FEN), Budget(max_transitions=1500))
    assert d.extra["score_bounded"] is False
    assert d.extra["root_bounds"] == (-1.0, 1.0)  # limites nunca atualizados


# --- needRecalc ---------------------------------------------------------------
def test_need_recalc_does_not_change_search_but_saves_sorts():
    state = initial_state(START_FEN)
    on = GPNMCTSAgent(seed=3, need_recalc=True).choose_action(
        state, Budget(max_transitions=6000))
    off = GPNMCTSAgent(seed=3, need_recalc=False).choose_action(
        state, Budget(max_transitions=6000))
    assert on.action.uci() == off.action.uci()
    for key in ("iterations", "playouts", "playout_steps", "proof_updates",
                "tree_nodes_created", "root_bounds"):
        assert on.extra[key] == off.extra[key]
    assert on.extra["bias_cache_hits"] > 0
    assert off.extra["bias_cache_hits"] == 0
    assert on.extra["pnrank_sorts"] < off.extra["pnrank_sorts"]


# --- Métricas e mobility init -------------------------------------------------
def test_pnrank_sorts_only_counted_for_rank():
    state = initial_state(START_FEN)
    rank = GPNMCTSAgent(seed=0, formula="rank").choose_action(
        state, Budget(max_transitions=3000))
    mx = GPNMCTSAgent(seed=0, formula="max").choose_action(
        state, Budget(max_transitions=3000))
    assert rank.extra["pnrank_sorts"] > 0
    assert mx.extra["pnrank_sorts"] == 0


def test_bias_counts_sort_in_stats():
    stats = GStats()
    _bias([1.0, 4.0, 9.0], "rank", stats)
    assert stats.pnrank_sorts == 1


def test_mobility_init_gives_and_player_the_move_count():
    state = initial_state(START_FEN)  # brancas a mover
    node = GNode(state, mobility_init=True)
    assert node.pn[chess.WHITE] == 1.0  # OR
    assert node.pn[chess.BLACK] == float(len(legal_actions(state)))  # AND


def test_mobility_init_off_by_default():
    assert GPNMCTSAgent().mobility_init is False


def test_mobility_init_agent_runs_and_reports():
    d = GPNMCTSAgent(seed=0, mobility_init=True).choose_action(
        initial_state(START_FEN), Budget(max_transitions=2000))
    assert d.extra["mobility_init"] is True
    assert d.action.uci() in {a.uci() for a in legal_actions(initial_state(START_FEN))}


# --- Contrato único (Seção 12.2) e orçamento por tempo --------------------------
def test_time_budget_is_respected_and_reported():
    state = initial_state(START_FEN)
    d = GPNMCTSAgent(seed=2026).choose_action(state, Budget(max_time_ms=50))
    assert d.action.uci() in {a.uci() for a in legal_actions(state)}
    assert d.extra["elapsed_ms"] < 50 + 100
    assert d.transitions > 0


def test_decision_exposes_common_metrics():
    d = GPNMCTSAgent(seed=2026).choose_action(initial_state(START_FEN),
                                              Budget(max_transitions=1000))
    assert d.transitions <= 1000
    assert d.nodes_visited >= d.nodes_expanded > 0
    for key in ("tree_nodes_created", "elapsed_ms", "cpu_ms", "budget_overrun"):
        assert key in d.extra
    assert d.extra["budget_overrun"] is False


def test_final_choice_is_most_visited_when_root_unsolved():
    # Plano 7.2: escolha final = filho com mais visitas (sem filtro de limites)
    state = initial_state(START_FEN)
    for score_bounded in (True, False):
        d = GPNMCTSAgent(seed=2026, score_bounded=score_bounded).choose_action(
            state, Budget(max_transitions=1500))
        assert not d.extra["root_solved"]
