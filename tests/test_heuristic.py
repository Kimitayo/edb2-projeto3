"""Testes da heurística compartilhada do projeto EDB II."""

import chess
import pytest

from game.evaluation import evaluate, score


def test_initial_position_has_zero_evaluation():
    board = chess.Board()

    assert score(board, chess.WHITE) == 0
    assert score(board, chess.BLACK) == 0
    assert evaluate(board, chess.WHITE) == 0
    assert evaluate(board, chess.BLACK) == 0


def test_evaluation_is_symmetric_between_players():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    white_value = evaluate(board, chess.WHITE)
    black_value = evaluate(board, chess.BLACK)

    assert white_value == pytest.approx(-black_value)


def test_non_terminal_evaluation_is_within_limits():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    for player in (chess.WHITE, chess.BLACK):
        value = evaluate(board, player)
        assert -0.99 < value < 0.99


def test_material_advantage_is_reflected_in_score():
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")

    assert score(board, chess.WHITE) > 0
    assert score(board, chess.BLACK) < 0
    assert evaluate(board, chess.WHITE) > 0
    assert evaluate(board, chess.BLACK) < 0


def test_evaluation_does_not_modify_the_board():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    fen_before = board.fen()
    move_stack_before = list(board.move_stack)

    evaluate(board, chess.WHITE)
    evaluate(board, chess.BLACK)

    assert board.fen() == fen_before
    assert board.move_stack == move_stack_before