
from __future__ import annotations # pra pdoer ligar classes mesmo que ainda não estejam definidas
import chess
from game import legal_actions


# Valores definidos que vão associar cada tipo de peça de xadrez a um valor material, ou seja, vai representar quanto cada peça vale na avaliação de posição
PIECE_VALUES = {
    chess.PAWN: 100, # peão
    chess.KNIGHT: 320, # cavalo
    chess.BISHOP: 330, # bispo
    chess.ROOK: 500, # torre
    chess.QUEEN: 900, # rainha
}

# Cria uma tabela de associação entre as caracterisiticas da partida e os 6 contextos que foi definido para a busca hibrida.
"""
OBJETIVO: pra analisr as melhores combinações e a busca híbrida apreder as melhores combinações

A ideia é combinar 2 informações:
1. Fase da partida: abertura, meio-jogo ou final
2. Situação da partida: tática ou não tática

# As seis combinações:
Contexto | Fase | Situação
C1 | Abertura | Não tática
C2 | Abertura | Tática
C3 | Meio-jogo | Não tática
C4 | Meio-jogo | Tática
C5 | Final | Não tática
C6 | Final | Tática

Então, por exemplo, se a função identificar:
```
phase = "midlegame"
situation = "tactical"
```
Aí consulta no dict context = CONTEXTS[(phase, situation)] e retorna C4

"""
CONTEXTS = {
    ("opening", "non_tactical"): "C1",
    ("opening", "tactical"): "C2",
    ("middlegame", "non_tactical"): "C3",
    ("middlegame", "tactical"): "C4",
    ("endgame", "non_tactical"): "C5",
    ("endgame", "tactical"): "C6",
}


# Serve para contar quantas peças tem no tabuleiro e já analisar se já está na fase final (endgame)
def count_non_king_non_pawn_pieces(board: chess.Board) -> int:
    """
    Conta cavalos, bispos, torres e damas dos dois jogadores.
    Reis e peões não entram na contagem.
    """
    piece_types = (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    )

    return sum(
        len(board.pieces(piece_type, color))
        for piece_type in piece_types
        for color in (chess.WHITE, chess.BLACK)
    )


def classify_phase(board: chess.Board) -> str:
    """
    Classifica a fase da partida.

    Retorna:
        opening: abertura
        middlegame: meio-jogo
        endgame: final

    A classificação de final tem prioridade sobre abertura.
    """
    non_king_non_pawn = count_non_king_non_pawn_pieces(board)

    if non_king_non_pawn <= 6:
        return "endgame"

    if board.fullmove_number <= 10:
        return "opening"

    return "middlegame"


def is_tactical(
    board: chess.Board,
    actions=None,
) -> bool:
    """
    Verifica se a posição é tática.

    Uma posição é tática quando:
    - o jogador da vez está em xeque; ou
    - existe pelo menos uma captura legal.
    """
    if board.is_check():
        return True

    if actions is None:
        actions = legal_actions(board)

    return any(
        isinstance(action, chess.Move) and board.is_capture(action)
        for action in actions
    )


def classify_situation(
    board: chess.Board,
    actions=None,
) -> str:
    """
    Classifica a situação como:
        tactical
        non_tactical
    """
    if is_tactical(board, actions):
        return "tactical"

    return "non_tactical"


def classify_context(
    board: chess.Board,
    actions=None,
) -> str:
    """
    Combina fase e situação tática.

    Retorna um dos seis contextos: C1, C2, C3, C4, C5 ou C6.
    """
    phase = classify_phase(board)
    situation = classify_situation(board, actions)

    return CONTEXTS[(phase, situation)]


def branching_factor(
    board: chess.Board,
    actions=None,
) -> int:
    """
    Calcula o fator de ramificação.

    Corresponde à quantidade de ações legais disponíveis,
    incluindo uma eventual reivindicação válida de empate.
    """
    if actions is None:
        actions = legal_actions(board)

    return len(actions)


def material_balance_cp(board: chess.Board) -> int:
    """
    Calcula o balanço material em centipeões.

    Convenção:
        resultado positivo: vantagem das brancas;
        resultado negativo: vantagem das pretas;
        zero: equilíbrio material.

    Utiliza os valores materiais definidos no projeto.
    """
    white_material = sum(
        len(board.pieces(piece_type, chess.WHITE)) * value
        for piece_type, value in PIECE_VALUES.items()
    )

    black_material = sum(
        len(board.pieces(piece_type, chess.BLACK)) * value
        for piece_type, value in PIECE_VALUES.items()
    )

    return white_material - black_material


def state_features(
    board: chess.Board,
    actions=None,
) -> dict:
    """
    Extrai todas as características utilizadas pela busca híbrida.

    Retorna:
        phase
        tactical
        situation
        context
        branching_factor
        material_balance_cp
        material_difference_cp
        in_check
        non_king_non_pawn_pieces
    """
    if actions is None:
        actions = legal_actions(board)

    phase = classify_phase(board)
    tactical = is_tactical(board, actions)
    situation = "tactical" if tactical else "non_tactical"

    balance = material_balance_cp(board)

    return {
        "phase": phase,
        "tactical": tactical,
        "situation": situation,
        "context": CONTEXTS[(phase, situation)],
        "branching_factor": branching_factor(board, actions),
        "material_balance_cp": balance,
        "material_difference_cp": abs(balance),
        "in_check": board.is_check(),
        "non_king_non_pawn_pieces": (
            count_non_king_non_pawn_pieces(board)
        ),
    }
