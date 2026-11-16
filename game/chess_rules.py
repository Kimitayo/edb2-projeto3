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
    render(state, unicode=True)          -> str   (tabuleiro em texto)
    render_svg(state, size=400)          -> str   (imagem SVG)
    serialize(state)                     -> dict  (compatível com JSON)
    deserialize(data)                    -> ChessState
    to_pgn(state)                        -> str
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
import chess.pgn
import chess.svg

__all__ = [
    "ClaimDraw", "Action", "Result", "Outcome", "ChessState",
    "initial_state", "from_moves", "legal_actions", "next_state",
    "is_terminal", "get_outcome", "utility", "render", "render_svg",
    "serialize", "deserialize", "to_pgn",
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

    @classmethod
    def from_uci(cls, text: str) -> "ClaimDraw":
        """Inverso de uci(): "claim" ou "claim:<lance UCI>"."""
        if text == "claim":
            return cls()
        prefix, sep, move = text.partition(":")
        if prefix != "claim" or not sep:
            raise ValueError(f"Reivindicação mal formada: {text!r}")
        return cls(chess.Move.from_uci(move))


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
    return _is_over(state)


def get_outcome(state: chess.Board) -> Outcome:
    """Vitória das brancas, das pretas, empate ou em andamento.

    O mate tem precedência sobre os empates automáticos (ex.: um mate no
    150º meio-lance vale como mate, não como regra dos 75)."""
    if getattr(state, "claimed_draw", None) is not None:
        return Outcome(Result.DRAW, "claimed_draw")
    o = state.outcome(claim_draw=False)
    if o is None:
        return Outcome(Result.ONGOING)
    if o.winner is chess.WHITE:
        result = Result.WHITE_WINS
    elif o.winner is chess.BLACK:
        result = Result.BLACK_WINS
    else:
        result = Result.DRAW
    return Outcome(result, o.termination.name.lower())


def utility(state: chess.Board, player: chess.Color) -> float:
    """+1 (vitória), 0 (empate) ou -1 (derrota) para `player`.
    Só definida em estados terminais (ValueError caso contrário)."""
    # Exige bool: 0/1 seriam aceitos por igualdade (0 == False == BLACK),
    # mas no código antigo 0 era o primeiro jogador (brancas).
    if not isinstance(player, bool):
        raise ValueError(f"Jogador deve ser chess.WHITE ou chess.BLACK, não {player!r}")
    if not _is_over(state):
        raise ValueError("utility() só é definida para estados terminais")
    winner = get_outcome(state).winner
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


# --- Depuração e registro ---------------------------------------------------
_RESULT_TEXT = {
    Result.WHITE_WINS: "vitória das brancas",
    Result.BLACK_WINS: "vitória das pretas",
    Result.DRAW: "empate",
}


def render(state: chess.Board, unicode: bool = True) -> str:
    """Tabuleiro em texto (brancas embaixo), com coordenadas e uma linha de
    informações: vez, lance, roque, en passant, contador dos 50 lances,
    último lance, xeque e resultado.

    unicode=False usa letras (K, q, ...) para terminais sem UTF-8.
    Obs.: em terminais de fundo escuro os símbolos Unicode das peças
    brancas e pretas podem parecer trocados; as letras não têm esse problema.
    """
    empty = "·" if unicode else "."
    lines = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = state.piece_at(chess.square(file, rank))
            if piece is None:
                cells.append(empty)
            else:
                cells.append(piece.unicode_symbol() if unicode else piece.symbol())
        lines.append(f"{rank + 1} {' '.join(cells)} {rank + 1}")
    lines.append("  a b c d e f g h")

    ep = (chess.square_name(state.ep_square)
          if state.ep_square is not None and state.has_legal_en_passant() else "-")
    info = [
        f"Vez: {'brancas' if state.turn == chess.WHITE else 'pretas'}",
        f"Lance {state.fullmove_number}",
        f"Roque: {state.castling_xfen()}",
        f"En passant: {ep}",
        f"Meio-lances s/ captura ou peão: {state.halfmove_clock}",
    ]
    if state.move_stack:
        info.append(f"Último: {state.peek().uci()}")
    lines.append(" | ".join(info))

    outcome = get_outcome(state)
    if outcome.result is not Result.ONGOING:
        claim = getattr(state, "claimed_draw", None)
        detail = outcome.reason if claim is None else f"{outcome.reason}: {claim.uci()}"
        lines.append(f"FIM: {_RESULT_TEXT[outcome.result]} ({detail})")
    elif state.is_check():
        lines.append("XEQUE!")
    return "\n".join(lines)


def render_svg(state: chess.Board, size: int = 400) -> str:
    """Imagem SVG do tabuleiro: seta no último lance e rei em xeque em
    vermelho. Abra o arquivo .svg no navegador ou no VS Code."""
    lastmove = state.peek() if state.move_stack else None
    check = state.king(state.turn) if state.is_check() else None
    return chess.svg.board(state, size=size, lastmove=lastmove, check=check)


def serialize(state: chess.Board) -> dict:
    """Dicionário (compatível com JSON) com FEN inicial, lances UCI, FEN
    atual, reivindicação e resultado — suficiente para reconstruir o estado
    com todo o histórico (necessário para detectar repetições)."""
    claim = getattr(state, "claimed_draw", None)
    outcome = get_outcome(state)
    return {
        "start_fen": state.root().fen(),
        "moves": [m.uci() for m in state.move_stack],
        "fen": state.fen(),
        "claimed_draw": None if claim is None else claim.uci(),
        "result": outcome.result.value,
        "reason": outcome.reason,
    }


def deserialize(data: dict) -> ChessState:
    """Inverso de serialize(). Reaplica os lances a partir de `start_fen`.
    ValueError se algum lance for ilegal, se a FEN final não bater com
    `fen` (quando presente) ou se a reivindicação registrada for inválida."""
    state = from_moves(data.get("moves", []), data.get("start_fen"))
    if data.get("fen") is not None and state.fen() != data["fen"]:
        raise ValueError(f"FEN final não confere: {state.fen()} != {data['fen']}")
    if data.get("claimed_draw") is not None:
        claim = ClaimDraw.from_uci(data["claimed_draw"])
        if not _claim_is_valid(state, claim):
            raise ValueError(f"Reivindicação inválida {claim.uci()} em {state.fen()}")
        state.claimed_draw = claim
    return state


def to_pgn(state: chess.Board, headers: Optional[dict] = None) -> str:
    """Partida em PGN, com o resultado do motor (inclui empate reivindicado,
    registrado também como comentário no fim)."""
    game_pgn = chess.pgn.Game.from_board(state)
    game_pgn.headers["Result"] = get_outcome(state).result.value
    for key, value in (headers or {}).items():
        game_pgn.headers[key] = str(value)
    claim = getattr(state, "claimed_draw", None)
    if claim is not None:
        game_pgn.end().comment = f"Empate reivindicado ({claim.uci()})"
    return str(game_pgn)
