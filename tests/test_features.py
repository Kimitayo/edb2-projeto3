
import chess

from game.features import (
    classify_phase,
    classify_context,
    is_tactical,
    branching_factor,
    material_balance_cp,
    state_features,
    count_non_king_non_pawn_pieces,
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


def test_opening_tactical_context():
    board = chess.Board(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    )

    assert classify_phase(board) == "opening"
    assert is_tactical(board) is True
    assert classify_context(board) == "C2"


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

def test_middlegame_non_tactical_context():
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 11"
    )

    assert classify_phase(board) == "middlegame"
    assert is_tactical(board) is False
    assert classify_context(board) == "C3"

def test_middlegame_tactical_context():
    board = chess.Board(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 11"
    )

    assert classify_phase(board) == "middlegame"
    assert is_tactical(board) is True
    assert classify_context(board) == "C4"

def test_endgame_non_tactical_context():
    board = chess.Board(
        "8/8/8/4k3/8/8/8/4K2R w - - 0 1"
    )

    assert classify_phase(board) == "endgame"
    assert is_tactical(board) is False
    assert classify_context(board) == "C5"

def test_endgame_tactical_context():
    board = chess.Board(
        "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"
    )

    assert classify_phase(board) == "endgame"
    assert is_tactical(board) is True
    assert classify_context(board) == "C6"

def test_six_non_king_non_pawn_pieces_is_endgame():
    board = chess.Board(
        "4k3/8/8/8/8/8/8/1NNBKRRR w - - 0 11"
    )

    assert count_non_king_non_pawn_pieces(board) == 6
    assert classify_phase(board) == "endgame"


def test_seven_non_king_non_pawn_pieces_is_middlegame():
    board = chess.Board(
        "3qk3/8/8/8/8/8/8/1NNBKRRR w - - 0 11"
    )

    assert count_non_king_non_pawn_pieces(board) == 7
    assert classify_phase(board) == "middlegame"

def test_tenth_move_is_opening():
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 10"
    )

    assert board.fullmove_number == 10
    assert classify_phase(board) == "opening"
    assert classify_context(board) == "C1"


def test_eleventh_move_is_middlegame():
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 11"
    )

    assert board.fullmove_number == 11
    assert classify_phase(board) == "middlegame"
    assert classify_context(board) == "C3"

def test_tactical_position_with_legal_capture_not_in_check():
    board = chess.Board(
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 11"
    )

    capture = chess.Move.from_uci("e4d5")

    assert board.is_check() is False
    assert capture in board.legal_moves
    assert board.is_capture(capture)
    assert is_tactical(board) is True

def test_non_tactical_position_without_check_or_captures():
    board = chess.Board()

    assert board.is_check() is False

    assert not any(
        board.is_capture(move)
        for move in board.legal_moves
    )

    assert is_tactical(board) is False

def test_state_features_returns_all_expected_fields():
    board = chess.Board(
        "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    )

    features = state_features(board)

    assert features["phase"] == "endgame"
    assert features["tactical"] is False
    assert features["situation"] == "non_tactical"
    assert features["context"] == "C5"

    assert features["branching_factor"] == branching_factor(board)

    assert features["material_balance_cp"] == 900
    assert features["material_difference_cp"] == 900

    assert features["in_check"] is False
    assert features["non_king_non_pawn_pieces"] == 1