"""Motor de xadrez — INTERFACE COMUM usada por todos os algoritmos.

Especificação: docs/PLANO_PROJETO_XADREZ_EDBII.md, Seções 3 e 4.
Responsável: Integrante 1 (Matheus Bulhões).

Contrato público:
    initial_state(fen=None)              -> chess.Board
    from_moves(moves, start_fen=None)    -> chess.Board
    legal_actions(state)                 -> tuple[Action, ...]
    next_state(state, action)            -> chess.Board   (novo objeto)
    is_terminal(state)                   -> bool
    get_outcome(state)                   -> Outcome
    utility(state, player)               -> +1.0 / 0.0 / -1.0 (só em terminal)
    render(state)                        -> str
    serialize(state)                     -> dict
    deserialize(data)                    -> chess.Board
    reset_transition_count() / transition_count()

Convenções:
- Estado = chess.Board com histórico (move_stack). Nunca é modificado:
  next_state trabalha numa cópia feita com copy(stack=True).
- Jogador = chess.WHITE (True) ou chess.BLACK (False).
- Ação = chess.Move ou ClaimDraw (reivindicação de empate).
- Transição = uma chamada bem-sucedida a next_state. O contador é global e
  serve de métrica comum de custo (Budget.max_transitions).

A heurística (evaluate) fica em game/evaluation.py (Integrante 2) e as
características da híbrida em game/features.py (Integrante 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Union

import chess

__all__ = [
    "ClaimDraw", "Action", "Result", "Outcome",
    "initial_state", "from_moves", "legal_actions", "next_state",
    "is_terminal", "get_outcome", "utility", "render",
    "serialize", "deserialize",
    "reset_transition_count", "transition_count",
]


# --- Ações ------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimDraw:
    """Reivindicação de empate (3ª repetição ou regra dos 50 movimentos).

    move = None    -> a posição ATUAL já permite reivindicar.
    move = m       -> reivindicação baseada no lance pretendido `m`: o lance
                      fica registrado, mas NÃO é executado no tabuleiro.
    """
    move: Optional[chess.Move] = None

    def uci(self) -> str:
        return "claim" if self.move is None else f"claim:{self.move.uci()}"


Action = Union[chess.Move, ClaimDraw]


# --- Resultado --------------------------------------------------------------
class Result(Enum):
    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    ONGOING = "*"


@dataclass(frozen=True)
class Outcome:
    """Resultado de um estado.

    reason: motivo do término, ex.: "checkmate", "stalemate",
    "insufficient_material", "seventyfive_moves", "fivefold_repetition",
    "claimed_draw". None quando a partida está em andamento.
    """
    result: Result
    reason: Optional[str] = None

    @property
    def winner(self) -> Optional[chess.Color]:
        if self.result is Result.WHITE_WINS:
            return chess.WHITE
        if self.result is Result.BLACK_WINS:
            return chess.BLACK
        return None


# --- Contador de transições (métrica comum de custo) ------------------------
_transitions = 0


def reset_transition_count() -> None:
    global _transitions
    _transitions = 0


def transition_count() -> int:
    return _transitions


# --- Criação de estados -----------------------------------------------------
def initial_state(fen: Optional[str] = None) -> chess.Board:
    """Posição inicial padrão, ou a posição `fen` (sem histórico anterior)."""
    raise NotImplementedError("Etapa 2")


def from_moves(moves: Iterable[Union[str, chess.Move]],
               start_fen: Optional[str] = None) -> chess.Board:
    """Reconstrói um estado aplicando `moves` (UCI ou chess.Move) a partir de
    `start_fen` (ou da posição inicial), preservando o histórico."""
    raise NotImplementedError("Etapa 2")


# --- Regras -----------------------------------------------------------------
def legal_actions(state: chess.Board) -> tuple[Action, ...]:
    """Lances legais (ordem UCI) seguidos das reivindicações de empate válidas.
    Devolve () em estados terminais."""
    raise NotImplementedError("Etapa 3")


def next_state(state: chess.Board, action: Action) -> chess.Board:
    """Novo estado após `action`. Valida a legalidade (ValueError se ilegal),
    nunca modifica `state` e incrementa o contador de transições."""
    raise NotImplementedError("Etapa 2")


def is_terminal(state: chess.Board) -> bool:
    """True em mate, afogamento, material insuficiente, regra dos 75,
    5ª repetição ou reivindicação de empate aceita."""
    raise NotImplementedError("Etapa 4")


def get_outcome(state: chess.Board) -> Outcome:
    """Vitória das brancas, das pretas, empate ou em andamento."""
    raise NotImplementedError("Etapa 4")


def utility(state: chess.Board, player: chess.Color) -> float:
    """+1 (vitória), 0 (empate) ou -1 (derrota) para `player`.
    Só definida em estados terminais (ValueError caso contrário)."""
    raise NotImplementedError("Etapa 4")


# --- Depuração e registro ---------------------------------------------------
def render(state: chess.Board) -> str:
    """Tabuleiro em texto, com vez, roque, en passant e contadores."""
    raise NotImplementedError("Etapa 6")


def serialize(state: chess.Board) -> dict:
    """Dicionário com FEN inicial, lances UCI, FEN atual, reivindicação e
    resultado — suficiente para reconstruir o estado com o histórico."""
    raise NotImplementedError("Etapa 6")


def deserialize(data: dict) -> chess.Board:
    """Inverso de serialize()."""
    raise NotImplementedError("Etapa 6")
