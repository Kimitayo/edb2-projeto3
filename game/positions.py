"""Catálogo de posições FUNCIONAIS do motor (regras especiais e fins de jogo).

Usado pelos testes do motor e útil para depurar os algoritmos. NÃO é o
conjunto experimental de 120 posições do Lichess (responsabilidade do
Integrante 5, em experiments/).

Cada posição guarda a FEN inicial e a sequência de lances (UCI), para que o
histórico — e, portanto, a detecção de repetições — seja preservado.

    from game.positions import load, names
    s = load("tripla_repeticao")
    python -m game.positions          # mostra todas as posições
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .chess_rules import ChessState, Outcome, Result, from_moves

# Sequências reutilizadas
KNIGHT_SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"]   # volta à posição inicial
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]
SCHOLARS_MATE = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]

# FENs reutilizadas
CASTLING_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
PROMOTION_FEN = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1"
# Pretas (Rg8) contra Rg6 + Ta1; pretas só têm Kf8 e Kh8.
FIFTY_FEN = "6k1/8/6K1/8/8/8/8/R7 b - - {clock} 80"
# Mesma posição, brancas a jogar com 149 meios-lances: Ta8 é mate, Ta2 não.
SEVENTYFIVE_FEN = "6k1/8/6K1/8/8/8/8/R7 w - - 149 100"


@dataclass(frozen=True)
class Position:
    description: str
    start_fen: Optional[str] = None            # None = posição inicial padrão
    moves: tuple[str, ...] = field(default_factory=tuple)
    outcome: Outcome = Outcome(Result.ONGOING)  # resultado esperado do motor


def _p(description, start_fen=None, moves=(), result=Result.ONGOING, reason=None):
    return Position(description, start_fen, tuple(moves), Outcome(result, reason))


POSITIONS: dict[str, Position] = {
    "inicial": _p("Posição inicial padrão (20 lances legais)"),
    "mate_do_louco": _p("Mate mais rápido possível: pretas vencem",
                        moves=FOOLS_MATE, result=Result.BLACK_WINS, reason="checkmate"),
    "mate_do_pastor": _p("Mate do pastor: brancas vencem",
                         moves=SCHOLARS_MATE, result=Result.WHITE_WINS, reason="checkmate"),
    "mate_em_1": _p("Brancas dão mate em 1 com Ta8", "6k1/8/6K1/8/8/8/8/R7 w - - 0 1"),
    "afogamento": _p("Pretas sem lances e sem xeque",
                     "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", result=Result.DRAW, reason="stalemate"),
    "material_insuficiente": _p("Rei e bispo contra rei",
                                "8/8/8/4k3/8/8/8/4KB2 w - - 0 1",
                                result=Result.DRAW, reason="insufficient_material"),
    "roques": _p("Os quatro roques disponíveis", CASTLING_FEN),
    "roque_sob_ataque": _p("f1 atacada: só o roque grande é permitido",
                           "4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1"),
    "en_passant": _p("Brancas podem capturar en passant em d6",
                     moves=["e2e4", "a7a6", "e4e5", "d7d5"]),
    "promocao": _p("Peão em e7 promove para Q, R, B ou N", PROMOTION_FEN),
    "promocao_com_captura": _p("Peão em e7 promove capturando em d8",
                               "3r4/4P3/8/8/8/8/k7/4K3 w - - 0 1"),
    "peca_cravada": _p("Bispo em e2 cravado pela torre em e7",
                       "4k3/4r3/8/8/8/8/4B3/4K3 w - - 0 1"),
    "xeque": _p("Brancas em xeque pela 1ª fileira", "4k3/8/8/8/8/8/8/r3K3 w - - 0 1"),
    "tripla_repeticao": _p("Posição inicial pela 3ª vez: empate reivindicável",
                           moves=KNIGHT_SHUFFLE * 2),
    "tripla_repeticao_com_lance": _p("O lance f6g8 repetiria a posição pela 3ª vez",
                                     moves=(KNIGHT_SHUFFLE * 2)[:-1]),
    "cinquenta_lances": _p("100 meios-lances sem captura/peão: empate reivindicável",
                           FIFTY_FEN.format(clock=100)),
    "cinquenta_lances_com_lance": _p("Qualquer lance completa os 50 lances",
                                     FIFTY_FEN.format(clock=99)),
    "setenta_e_cinco_lances": _p("150 meios-lances: empate automático",
                                 SEVENTYFIVE_FEN, ["a1a2"],
                                 result=Result.DRAW, reason="seventyfive_moves"),
    "mate_no_lance_75": _p("Mate no 150º meio-lance: o mate prevalece",
                           SEVENTYFIVE_FEN, ["a1a8"],
                           result=Result.WHITE_WINS, reason="checkmate"),
    "quintupla_repeticao": _p("Posição inicial pela 5ª vez: empate automático",
                              moves=KNIGHT_SHUFFLE * 4,
                              result=Result.DRAW, reason="fivefold_repetition"),
}


def names() -> list[str]:
    return list(POSITIONS)


def load(name: str) -> ChessState:
    """Estado da posição `name`, com o histórico completo."""
    try:
        p = POSITIONS[name]
    except KeyError:
        raise KeyError(f"Posição desconhecida {name!r}. Opções: {', '.join(POSITIONS)}") from None
    return from_moves(p.moves, p.start_fen)


if __name__ == "__main__":
    import sys
    from .chess_rules import render
    sys.stdout.reconfigure(encoding="utf-8")
    for name, p in POSITIONS.items():
        print(f"=== {name}: {p.description}")
        print(render(load(name)))
        print()
