import chess

from algorithms.base import Budget
from algorithms.minimax import MinimaxAgent
from game.chess_rules import legal_actions


def test_mate_em_um():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    d = MinimaxAgent().choose_action(board, Budget(max_time_s=1.0))
    assert d.action.uci() == "a1a8"
    assert d.extra["root_value"] == 1.0
    assert d.extra["complete_depth"] == 1


def test_mate_em_dois():
    board = chess.Board("7k/8/8/8/8/8/1R6/R5K1 w - - 0 1")
    d = MinimaxAgent().choose_action(board, Budget(max_time_s=5.0))
    assert d.extra["root_value"] == 1.0
    assert d.extra["complete_depth"] <= 3


def test_estado_original_intacto_e_acao_legal():
    board = chess.Board()
    fen, stack = board.fen(), list(board.move_stack)
    d = MinimaxAgent().choose_action(board, Budget(max_time_s=0.1))
    assert board.fen() == fen and list(board.move_stack) == stack
    assert d.action in legal_actions(board)


def test_alfa_beta_igual_ao_minimax_sem_poda():
    board = chess.Board()
    a = MinimaxAgent(use_pruning=True).choose_action(board, Budget(max_depth=2))
    b = MinimaxAgent(use_pruning=False).choose_action(board, Budget(max_depth=2))
    assert a.extra["root_value"] == b.extra["root_value"]
    assert a.action == b.action
    assert a.nodes_visited <= b.nodes_visited


def test_deterministico():
    board = chess.Board()
    a = MinimaxAgent().choose_action(board, Budget(max_depth=3))
    b = MinimaxAgent().choose_action(board, Budget(max_depth=3))
    assert a.action == b.action


def test_orcamento_de_transicoes_respeitado():
    d = MinimaxAgent().choose_action(chess.Board(), Budget(max_transitions=500))
    assert d.extra["transitions"] <= 500


def test_orcamento_minusculo_retorna_primeira_uci():
    board = chess.Board()
    d = MinimaxAgent().choose_action(board, Budget(max_transitions=1))
    assert d.action.uci() == min(m.uci() for m in board.legal_moves)
    assert d.extra["complete_depth"] == 0
