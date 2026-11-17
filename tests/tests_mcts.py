import os
import sys

import chess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algorithms.mcts import choose_action, uct_search  # noqa: E402

# Mate em 1 para as brancas: Ra8#
MATE_BRANCAS = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
# Mate em 1 para as pretas: ...Ra1#
MATE_PRETAS = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


def test_retorna_lance_legal_na_posicao_inicial():
    board = chess.Board()
    move = choose_action(board, time_budget=0.3, seed=1)
    assert move in board.legal_moves


def test_nao_altera_o_tabuleiro_do_chamador():
    board = chess.Board()
    antes = board.fen()
    choose_action(board, time_budget=0.2, seed=1)
    assert board.fen() == antes


def test_encontra_mate_em_1_brancas():
    board = chess.Board(MATE_BRANCAS)
    move = choose_action(board, time_budget=1.0, seed=42)
    assert move == chess.Move.from_uci("a1a8")


def test_encontra_mate_em_1_pretas():
    board = chess.Board(MATE_PRETAS)
    move = choose_action(board, time_budget=1.0, seed=42)
    assert move == chess.Move.from_uci("a8a1")


def test_posicao_terminal_levanta_erro():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # afogamento
    assert board.is_game_over()
    with pytest.raises(ValueError):
        choose_action(board, time_budget=0.1)


def test_reprodutivel_com_semente_e_iteracoes_fixas():
    board = chess.Board()
    r1 = uct_search(board, time_budget=60, seed=7, max_iterations=300)
    r2 = uct_search(board, time_budget=60, seed=7, max_iterations=300)
    assert r1.move == r2.move
    assert r1.root_visits == r2.root_visits


def test_respeita_o_orcamento_de_tempo():
    board = chess.Board()
    r = uct_search(board, time_budget=0.5, seed=1)
    assert 0.5 <= r.elapsed < 0.8
    assert r.iterations > 0 and r.nodes > 1


def test_visitas_na_raiz_somam_as_iteracoes():
    board = chess.Board()
    r = uct_search(board, time_budget=60, seed=3, max_iterations=200)
    assert r.iterations == 200
    assert sum(r.root_visits.values()) == 200