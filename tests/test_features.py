
import chess

from game.features import (
    classify_phase,
    classify_context,
    is_tactical,
    branching_factor,
    material_balance_cp,
    state_features,
)


def test_initial_position():
    board = chess.Board()

    features = state_features(board)

    assert features["phase"] == "opening"
    assert features["situation"] == "non_tactical"
    assert features["context"] == "C1"
    assert features["branching_factor"] == 20
    assert features["material_balance_cp"] == 0
    assert features["in_check"] is False


def test_endgame_classification():
    board = chess.Board(
        "8/8/8/4k3/8/8/8/4K2R w - - 0 1"
    )

    assert classify_phase(board) == "endgame"


def test_endgame_has_priority_over_opening():
    board = chess.Board(
        "8/8/8/4k3/8/8/8/4K2R w - - 0 1"
    )

    assert board.fullmove_number <= 10
    assert classify_phase(board) == "endgame"


def test_tactical_position_in_check():
    board = chess.Board(
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    )

    assert is_tactical(board)


def test_initial_context():
    assert classify_context(chess.Board()) == "C1"


def test_material_balance():
    board = chess.Board(
        "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    )

    assert material_balance_cp(board) == 900


def test_branching_factor():
    board = chess.Board()

    assert branching_factor(board) == len(
        list(board.legal_moves)
    )
