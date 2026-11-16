"""Testes do motor de xadrez (game/chess_rules.py)."""
import dataclasses

import chess
import pytest

import game
from game import (ChessState, ClaimDraw, Outcome, Result, from_moves,
                  initial_state, next_state)


# --- Etapa 1: contrato da interface -----------------------------------------
PUBLIC_API = [
    "initial_state", "from_moves", "legal_actions", "next_state",
    "is_terminal", "get_outcome", "utility", "render",
    "serialize", "deserialize",
    "reset_transition_count", "transition_count",
]


@pytest.mark.parametrize("name", PUBLIC_API)
def test_public_function_is_exported(name):
    assert callable(getattr(game, name))


def test_claim_draw_is_immutable_and_hashable():
    m = chess.Move.from_uci("g1f3")
    c = ClaimDraw(m)
    assert c == ClaimDraw(chess.Move.from_uci("g1f3"))
    assert hash(c) == hash(ClaimDraw(m))
    assert ClaimDraw() != c
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.move = None


def test_claim_draw_uci():
    assert ClaimDraw().uci() == "claim"
    assert ClaimDraw(chess.Move.from_uci("e2e4")).uci() == "claim:e2e4"


def test_outcome_winner():
    assert Outcome(Result.WHITE_WINS, "checkmate").winner == chess.WHITE
    assert Outcome(Result.BLACK_WINS, "checkmate").winner == chess.BLACK
    assert Outcome(Result.DRAW, "stalemate").winner is None
    assert Outcome(Result.ONGOING).winner is None


def test_transition_counter_reset():
    game.reset_transition_count()
    assert game.transition_count() == 0


# --- Etapa 2: estado e transição --------------------------------------------
E4 = chess.Move.from_uci("e2e4")
# Cavalos vão e voltam: a posição inicial se repete a cada 4 meios-lances.
KNIGHT_SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"]


def test_initial_state_is_standard_board():
    s = initial_state()
    assert isinstance(s, ChessState) and isinstance(s, chess.Board)
    assert s.fen() == chess.STARTING_FEN
    assert s.move_stack == []
    assert s.claimed_draw is None
    assert s.legal_moves.count() == 20


def test_initial_state_from_fen():
    fen = "8/8/8/4k3/8/8/8/4K2R w K - 0 1"
    assert initial_state(fen).fen() == fen


@pytest.mark.parametrize("fen", [
    "isto nao e uma fen",
    "8/8/8/8/8/8/8/8 w - - 0 1",            # sem reis
    "4k3/8/8/8/8/8/8/4K2r b - - 0 1",       # brancas em xeque com a vez das pretas
])
def test_initial_state_rejects_invalid_fen(fen):
    with pytest.raises(ValueError):
        initial_state(fen)


def test_from_moves_preserves_history():
    s = from_moves(["e2e4", chess.Move.from_uci("e7e5"), "g1f3"])
    assert [m.uci() for m in s.move_stack] == ["e2e4", "e7e5", "g1f3"]
    assert s.root().fen() == chess.STARTING_FEN
    assert s.turn == chess.BLACK


def test_from_moves_rejects_illegal_move():
    with pytest.raises(ValueError):
        from_moves(["e2e4", "e2e4"])


def test_from_moves_does_not_count_transitions():
    game.reset_transition_count()
    from_moves(["e2e4", "e7e5"])
    assert game.transition_count() == 0


def test_next_state_applies_move():
    s = initial_state()
    child = next_state(s, E4)
    assert isinstance(child, ChessState)
    assert child.move_stack == [E4]
    assert child.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
    assert child.turn == chess.BLACK


def test_next_state_does_not_modify_original():
    s = from_moves(["e2e4", "e7e5"])
    fen, stack = s.fen(), list(s.move_stack)
    child = next_state(s, chess.Move.from_uci("g1f3"))
    assert child is not s
    assert s.fen() == fen and s.move_stack == stack
    assert child.move_stack == stack + [chess.Move.from_uci("g1f3")]


def test_next_state_keeps_repetition_history():
    s = from_moves(KNIGHT_SHUFFLE[:3])
    child = next_state(s, chess.Move.from_uci("f6g8"))  # volta à posição inicial
    assert child.is_repetition(2)


@pytest.mark.parametrize("uci", [
    "e2e5",    # peão não anda 3 casas
    "e7e5",    # peça do adversário
    "0000",    # lance nulo
])
def test_next_state_rejects_illegal_move(uci):
    game.reset_transition_count()
    with pytest.raises(ValueError):
        next_state(initial_state(), chess.Move.from_uci(uci))
    assert game.transition_count() == 0


def test_promotion_requires_piece():
    s = initial_state("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")
    with pytest.raises(ValueError):
        next_state(s, chess.Move.from_uci("e7e8"))       # sem a peça
    assert next_state(s, chess.Move.from_uci("e7e8q")).piece_at(chess.E8).symbol() == "Q"


def test_next_state_rejects_non_action():
    with pytest.raises(TypeError):
        next_state(initial_state(), "e2e4")


def test_next_state_rejects_move_after_game_over():
    # 5ª repetição da posição inicial encerra a partida automaticamente.
    s = from_moves(KNIGHT_SHUFFLE * 4)
    assert s.is_fivefold_repetition()
    with pytest.raises(ValueError):
        next_state(s, chess.Move.from_uci("g1f3"))


def test_next_state_counts_transitions():
    game.reset_transition_count()
    s = initial_state()
    for uci in ["e2e4", "e7e5", "g1f3"]:
        s = next_state(s, chess.Move.from_uci(uci))
    assert game.transition_count() == 3


def test_next_state_accepts_plain_board():
    board = chess.Board()
    board.push_uci("e2e4")
    child = next_state(board, chess.Move.from_uci("e7e5"))
    assert isinstance(child, ChessState)
    assert [m.uci() for m in child.move_stack] == ["e2e4", "e7e5"]
    assert board.move_stack == [E4]


def test_copy_preserves_claimed_draw():
    s = initial_state()
    s.claimed_draw = ClaimDraw()
    assert s.copy().claimed_draw == ClaimDraw()
    assert s.copy(stack=False).claimed_draw == ClaimDraw()


def test_claim_draw_not_yet_supported():
    with pytest.raises(NotImplementedError):
        next_state(initial_state(), ClaimDraw())
