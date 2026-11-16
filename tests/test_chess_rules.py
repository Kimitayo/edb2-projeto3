"""Testes do motor de xadrez (game/chess_rules.py)."""
import dataclasses

import chess
import pytest

import game
from game import (ChessState, ClaimDraw, Outcome, Result, from_moves,
                  get_outcome, initial_state, is_terminal, legal_actions,
                  next_state, utility)


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


# --- Etapa 3: ações legais e reivindicação de empate ------------------------
def mv(uci):
    return chess.Move.from_uci(uci)


def claims(actions):
    return [a for a in actions if isinstance(a, ClaimDraw)]


# Rei preto em g8; brancas com rei g6 e torre a1. Pretas só têm Kf8 e Kh8.
FIFTY_FEN = "6k1/8/6K1/8/8/8/8/R7 b - - {clock} 80"


def test_legal_actions_initial_position():
    actions = legal_actions(initial_state())
    assert len(actions) == 20
    assert all(isinstance(a, chess.Move) for a in actions)
    assert [a.uci() for a in actions] == sorted(a.uci() for a in actions)


def test_legal_actions_are_exactly_the_legal_moves():
    s = from_moves(["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"])
    moves = [a for a in legal_actions(s) if isinstance(a, chess.Move)]
    assert set(moves) == set(s.legal_moves)
    assert claims(legal_actions(s)) == []


def test_threefold_repetition_claim_now():
    s = from_moves(KNIGHT_SHUFFLE * 2)            # posição inicial pela 3ª vez
    assert claims(legal_actions(s)) == [ClaimDraw()]
    assert legal_actions(s)[-1] == ClaimDraw()     # reivindicações vêm depois dos lances


def test_threefold_repetition_claim_with_intended_move():
    s = from_moves((KNIGHT_SHUFFLE * 2)[:-1])     # f6g8 repetiria pela 3ª vez
    assert claims(legal_actions(s)) == [ClaimDraw(mv("f6g8"))]


def test_fifty_moves_claim_now():
    s = initial_state(FIFTY_FEN.format(clock=100))
    assert claims(legal_actions(s)) == [ClaimDraw()]


def test_fifty_moves_claim_with_intended_move():
    s = initial_state(FIFTY_FEN.format(clock=99))
    assert claims(legal_actions(s)) == [ClaimDraw(mv("g8f8")), ClaimDraw(mv("g8h8"))]


def test_no_claim_below_fifty_moves():
    s = initial_state(FIFTY_FEN.format(clock=98))
    assert claims(legal_actions(s)) == []


def test_claim_draw_ends_game_without_playing_the_move():
    s = from_moves((KNIGHT_SHUFFLE * 2)[:-1])
    game.reset_transition_count()
    child = next_state(s, ClaimDraw(mv("f6g8")))
    assert child.claimed_draw == ClaimDraw(mv("f6g8"))
    assert child.move_stack == s.move_stack           # lance pretendido não executado
    assert child.fen() == s.fen()
    assert s.claimed_draw is None                      # original intacto
    assert legal_actions(child) == ()
    assert game.transition_count() == 1


@pytest.mark.parametrize("claim", [
    ClaimDraw(),                    # posição inicial: nada a reivindicar
    ClaimDraw(mv("g1f3")),          # lance que não gera repetição
    ClaimDraw(mv("e2e5")),          # lance ilegal
])
def test_invalid_claim_is_rejected(claim):
    with pytest.raises(ValueError):
        next_state(initial_state(), claim)


def test_no_actions_after_claimed_draw():
    s = next_state(from_moves(KNIGHT_SHUFFLE * 2), ClaimDraw())
    with pytest.raises(ValueError):
        next_state(s, mv("g1f3"))
    with pytest.raises(ValueError):
        next_state(s, ClaimDraw())


# --- Etapa 4: fim de jogo e utilidade ---------------------------------------
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]          # pretas vencem
SCHOLARS_MATE = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
# Brancas: Rg6 + Ta1 contra Rg8. Ta8 é mate; Ta2 não é.
SEVENTYFIVE_FEN = "6k1/8/6K1/8/8/8/8/R7 w - - 149 100"


def assert_terminal(s, result, reason):
    assert is_terminal(s)
    assert get_outcome(s) == Outcome(result, reason)
    assert legal_actions(s) == ()


def test_initial_position_is_ongoing():
    s = initial_state()
    assert not is_terminal(s)
    assert get_outcome(s) == Outcome(Result.ONGOING, None)
    with pytest.raises(ValueError):
        utility(s, chess.WHITE)


def test_checkmate_black_wins():
    s = from_moves(FOOLS_MATE)
    assert_terminal(s, Result.BLACK_WINS, "checkmate")
    assert utility(s, chess.BLACK) == 1.0
    assert utility(s, chess.WHITE) == -1.0


def test_checkmate_white_wins():
    s = from_moves(SCHOLARS_MATE)
    assert_terminal(s, Result.WHITE_WINS, "checkmate")
    assert utility(s, chess.WHITE) == 1.0
    assert utility(s, chess.BLACK) == -1.0


def test_stalemate():
    s = initial_state("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert_terminal(s, Result.DRAW, "stalemate")
    assert utility(s, chess.WHITE) == utility(s, chess.BLACK) == 0.0


@pytest.mark.parametrize("fen", [
    "8/8/8/4k3/8/8/8/4K3 w - - 0 1",        # K x K
    "8/8/8/4k3/8/8/8/4KB2 w - - 0 1",       # K+B x K
    "8/8/8/4k3/8/8/8/4KN2 w - - 0 1",       # K+N x K
])
def test_insufficient_material(fen):
    assert_terminal(initial_state(fen), Result.DRAW, "insufficient_material")


def test_sufficient_material_is_not_terminal():
    assert not is_terminal(initial_state("8/8/8/4k3/8/8/8/4K2R w - - 0 1"))


def test_seventyfive_moves_rule():
    s = next_state(initial_state(SEVENTYFIVE_FEN), mv("a1a2"))
    assert s.halfmove_clock == 150
    assert_terminal(s, Result.DRAW, "seventyfive_moves")


def test_checkmate_has_precedence_over_seventyfive_moves():
    s = next_state(initial_state(SEVENTYFIVE_FEN), mv("a1a8"))
    assert s.halfmove_clock == 150
    assert_terminal(s, Result.WHITE_WINS, "checkmate")


def test_fivefold_repetition_is_automatic():
    s = from_moves(KNIGHT_SHUFFLE * 4)
    assert_terminal(s, Result.DRAW, "fivefold_repetition")


def test_threefold_repetition_is_not_automatic():
    s = from_moves(KNIGHT_SHUFFLE * 2)
    assert not is_terminal(s)          # só termina se alguém reivindicar


def test_claimed_draw_outcome():
    s = next_state(from_moves(KNIGHT_SHUFFLE * 2), ClaimDraw())
    assert_terminal(s, Result.DRAW, "claimed_draw")
    assert utility(s, chess.WHITE) == 0.0


@pytest.mark.parametrize("player", [0, 1, "white", None])
def test_utility_rejects_invalid_player(player):
    with pytest.raises(ValueError):
        utility(from_moves(FOOLS_MATE), player)
