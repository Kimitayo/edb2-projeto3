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
- Estado = ChessState, subclasse de chess.Board com histórico (move_stack)
  e o campo `claimed_draw`. Nunca é modificado: next_state trabalha numa
  cópia feita com copy(stack=True).
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
    "ClaimDraw", "Action", "Result", "Outcome", "ChessState",
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


# --- Estado -----------------------------------------------------------------
class ChessState(chess.Board):
    """chess.Board com um campo a mais: a reivindicação de empate aceita.

    Continua sendo um chess.Board (isinstance funciona e toda a API da
    biblioteca está disponível). `claimed_draw` é None enquanto ninguém
    reivindicou empate; copy() o preserva.
    """

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN, *,
                 chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)
        self.claimed_draw: Optional[ClaimDraw] = None

    def copy(self, *, stack: Union[bool, int] = True) -> "ChessState":
        board = super().copy(stack=stack)
        board.claimed_draw = self.claimed_draw
        return board


def _as_state(board: chess.Board) -> ChessState:
    """Converte um chess.Board comum em ChessState, preservando o histórico."""
    if isinstance(board, ChessState):
        return board
    state = ChessState(board.root().fen())
    for move in board.move_stack:
        state.push(move)
    return state


def _is_over(state: chess.Board) -> bool:
    """Fim de jogo automático ou por reivindicação aceita."""
    return (getattr(state, "claimed_draw", None) is not None
            or state.is_game_over(claim_draw=False))


# --- Criação de estados -----------------------------------------------------
def initial_state(fen: Optional[str] = None) -> ChessState:
    """Posição inicial padrão, ou a posição `fen` (sem histórico anterior).
    ValueError se a FEN for inválida ou descrever uma posição ilegal."""
    state = ChessState(fen if fen is not None else chess.STARTING_FEN)
    if not state.is_valid():
        raise ValueError(f"Posição inválida: {fen!r} ({state.status()!r})")
    return state


def from_moves(moves: Iterable[Union[str, chess.Move]],
               start_fen: Optional[str] = None) -> ChessState:
    """Reconstrói um estado aplicando `moves` (UCI ou chess.Move) a partir de
    `start_fen` (ou da posição inicial), preservando o histórico.
    ValueError se algum lance for ilegal ou vier depois do fim da partida.
    Não conta transições (não é uma busca)."""
    state = initial_state(start_fen)
    for i, m in enumerate(moves):
        move = chess.Move.from_uci(m) if isinstance(m, str) else m
        if _is_over(state) or not state.is_legal(move):
            raise ValueError(f"Lance {i + 1} ({move.uci()}) ilegal em {state.fen()}")
        state.push(move)
    return state


# --- Reivindicação de empate ------------------------------------------------
# Uma repetição tripla exige ao menos 8 meios-lances reversíveis seguidos
# (posição em p, p+4 e p+8). Abaixo disso nenhum lance pode gerar uma
# reivindicação por repetição, e o teste lance a lance é pulado.
_MIN_CLOCK_FOR_REPETITION = 7
_FIFTY_MOVES_PLIES = 100


def _claimable_now(board: chess.Board) -> bool:
    """3ª repetição da posição atual ou 50 lances sem captura/movimento de peão."""
    return board.halfmove_clock >= _FIFTY_MOVES_PLIES or board.is_repetition(3)


def _claimable_after(board: chess.Board, move: chess.Move) -> bool:
    """O lance pretendido `move` (legal) tornaria a reivindicação válida?
    Um lance que dá mate não gera reivindicação (o jogador simplesmente vence)."""
    board.push(move)
    try:
        return _claimable_now(board) and not board.is_checkmate()
    finally:
        board.pop()


def _claim_is_valid(state: chess.Board, claim: ClaimDraw) -> bool:
    if _is_over(state):
        return False
    if claim.move is None:
        return _claimable_now(state)
    if not state.is_legal(claim.move):
        return False
    return _claimable_after(state.copy(stack=True), claim.move)


# --- Regras -----------------------------------------------------------------
def legal_actions(state: chess.Board) -> tuple[Action, ...]:
    """Lances legais (ordem UCI) seguidos das reivindicações de empate válidas.
    Devolve () em estados terminais.

    Reivindicações:
    - se a posição atual já permite reivindicar, só ClaimDraw() é incluída
      (reivindicações com lance pretendido levariam ao mesmo empate);
    - caso contrário, ClaimDraw(m) para cada lance m que tornaria a
      reivindicação válida (ordem UCI).
    """
    if _is_over(state):
        return ()
    moves = sorted(state.legal_moves, key=chess.Move.uci)
    if _claimable_now(state):
        return (*moves, ClaimDraw())
    if state.halfmove_clock < _MIN_CLOCK_FOR_REPETITION:  # atalho de desempenho
        return tuple(moves)
    probe = state.copy(stack=True)
    claims = [ClaimDraw(m) for m in moves if _claimable_after(probe, m)]
    return (*moves, *claims)


def next_state(state: chess.Board, action: Action) -> chess.Board:
    """Novo estado após `action`. Valida a legalidade (ValueError se ilegal),
    nunca modifica `state` e incrementa o contador de transições.
    Aceita chess.Board comum; o retorno é sempre um ChessState.

    ClaimDraw: o estado sucessor é a mesma posição, com `claimed_draw`
    preenchido (partida encerrada em empate). O lance pretendido NÃO é
    executado."""
    global _transitions
    if isinstance(action, ClaimDraw):
        if not _claim_is_valid(state, action):
            raise ValueError(f"Reivindicação inválida {action.uci()} em {state.fen()}")
        child = _as_state(state).copy(stack=True)
        child.claimed_draw = action
        _transitions += 1
        return child
    if not isinstance(action, chess.Move):
        raise TypeError(f"Ação deve ser chess.Move ou ClaimDraw, não {type(action).__name__}")
    if _is_over(state):
        raise ValueError(f"Partida encerrada; nenhuma ação é permitida em {state.fen()}")
    if not state.is_legal(action):
        raise ValueError(f"Lance ilegal {action.uci()} em {state.fen()}")
    child = _as_state(state).copy(stack=True)
    child.push(action)
    _transitions += 1
    return child


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
