"""Heurística compartilhada do projeto EDB II — xadrez.

Implementa exatamente os pesos e a normalização definidos na Seção 5
da especificação oficial do grupo (versão 2.0).

Interface pública:
    evaluate(state, player) -> float

A avaliação é sempre feita do ponto de vista de `player`:
valores positivos favorecem esse jogador; valores negativos favorecem
o adversário. Estados terminais devem ser tratados pelos algoritmos
com game.utility(), não por esta heurística.
"""
from __future__ import annotations

import math

import chess


# Valores materiais em centipeões (Seção 5.1).
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

CENTER_SQUARES = (
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
)

# Pesos posicionais (Seção 5.2).
WEIGHTS = {
    "mobility": 5,
    "center": 12,
    "doubled_pawns": -20,
    "isolated_pawns": -15,
    "passed_pawns": 30,
    "bishop_pair": 30,
    "king_risk": -20,
    "king_in_check": -40,
}


def _material(board: chess.Board, color: chess.Color) -> int:
    """Soma o material do jogador; o rei não recebe valor material."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def _mobility(board: chess.Board, color: chess.Color) -> int:
    """Conta ataques pseudo-legais de N/B/R/Q por peça.

    Casas ocupadas por peças próprias são excluídas. Peças cravadas
    continuam contando, conforme a especificação do projeto.
    """
    own_occupancy = board.occupied_co[color]
    total = 0

    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, color):
            attacked = board.attacks(square)
            total += (int(attacked) & ~own_occupancy).bit_count()

    return total


def _center_control(board: chess.Board, color: chess.Color) -> int:
    """Conta uma vez cada casa central atacada pelo jogador."""
    return sum(board.is_attacked_by(color, square) for square in CENTER_SQUARES)


def _doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    """Conta cada peão adicional existente em uma mesma coluna."""
    doubled = 0
    for file_index in range(8):
        count = sum(
            chess.square_file(square) == file_index
            for square in board.pieces(chess.PAWN, color)
        )
        doubled += max(0, count - 1)
    return doubled


def _isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    """Conta peões sem peões aliados nas colunas imediatamente vizinhas."""
    pawn_squares = board.pieces(chess.PAWN, color)
    pawn_files = {chess.square_file(square) for square in pawn_squares}
    isolated = 0

    for square in pawn_squares:
        file_index = chess.square_file(square)
        has_neighbor_file_pawn = (
            (file_index > 0 and file_index - 1 in pawn_files)
            or (file_index < 7 and file_index + 1 in pawn_files)
        )
        if not has_neighbor_file_pawn:
            isolated += 1

    return isolated


def _is_passed_pawn(
    board: chess.Board, square: chess.Square, color: chess.Color
) -> bool:
    """Verifica se não há peão adversário à frente na mesma/adjacente coluna."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for enemy_square in enemy_pawns:
        enemy_file = chess.square_file(enemy_square)
        enemy_rank = chess.square_rank(enemy_square)

        if abs(enemy_file - file_index) > 1:
            continue

        if color == chess.WHITE and enemy_rank > rank_index:
            return False
        if color == chess.BLACK and enemy_rank < rank_index:
            return False

    return True


def _passed_pawns(board: chess.Board, color: chess.Color) -> int:
    """Conta peões passados do jogador."""
    return sum(
        _is_passed_pawn(board, square, color)
        for square in board.pieces(chess.PAWN, color)
    )


def _bishop_pair(board: chess.Board, color: chess.Color) -> int:
    """Retorna 1 se o jogador possui pelo menos dois bispos; caso contrário, 0."""
    return int(len(board.pieces(chess.BISHOP, color)) >= 2)


def _king_risk(board: chess.Board, color: chess.Color) -> int:
    """Conta casas adjacentes ao rei atacadas pelo adversário.

    Cada casa é contada uma única vez, mesmo que seja atacada por várias
    peças adversárias. A contagem usa ataques pseudo-legais do tabuleiro.
    """
    king_square = board.king(color)
    if king_square is None:
        return 0

    adjacent_squares = board.attacks(king_square)
    risk = 0
    for square in adjacent_squares:
        if board.is_attacked_by(not color, square):
            risk += 1
    return risk


def _king_in_check(board: chess.Board, color: chess.Color) -> int:
    """Retorna 1 se o rei do jogador está em xeque; caso contrário, 0."""
    return int(board.is_check() and board.turn == color)


def _side_features(board: chess.Board, color: chess.Color) -> dict[str, int]:
    """Calcula as características de um único lado, antes da ponderação."""
    return {
        "material": _material(board, color),
        "mobility": _mobility(board, color),
        "center": _center_control(board, color),
        "doubled_pawns": _doubled_pawns(board, color),
        "isolated_pawns": _isolated_pawns(board, color),
        "passed_pawns": _passed_pawns(board, color),
        "bishop_pair": _bishop_pair(board, color),
        "king_risk": _king_risk(board, color),
        "king_in_check": _king_in_check(board, color),
    }


def score(state: chess.Board, player: chess.Color) -> float:
    """Calcula S(s,p), antes da normalização, em centipeões ponderados."""
    if player not in (chess.WHITE, chess.BLACK):
        raise ValueError("player deve ser chess.WHITE ou chess.BLACK")

    own = _side_features(state, player)
    opponent = _side_features(state, not player)

    material_delta = own["material"] - opponent["material"]
    score_value = material_delta

    for feature, weight in WEIGHTS.items():
        delta = own[feature] - opponent[feature]
        score_value += weight * delta

    return float(score_value)


def evaluate(state: chess.Board, player: chess.Color) -> float:
    """Retorna H(s,p) = 0.99 * tanh(S(s,p) / 600).

    A função é antissimétrica: evaluate(s, WHITE) == -evaluate(s, BLACK).
    Para estados terminais, os agentes devem usar game.utility().
    """
    raw_score = score(state, player)
    return 0.99 * math.tanh(raw_score / 600.0)


__all__ = [
    "PIECE_VALUES",
    "WEIGHTS",
    "score",
    "evaluate",
]
