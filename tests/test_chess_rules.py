"""Testes do motor de xadrez (game/chess_rules.py)."""
import dataclasses

import chess
import pytest

import game
from game import ClaimDraw, Outcome, Result


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
