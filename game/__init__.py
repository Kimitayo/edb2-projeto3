"""Motor de xadrez do projeto (Integrante 1).

Especificação: docs/PLANO_PROJETO_XADREZ_EDBII.md, Seção 3.
Uso: from game import legal_actions, next_state, is_terminal, ...
Demonstração: python -m game
"""
from .chess_rules import (ClaimDraw, Action, Result, Outcome, ChessState,
                          initial_state, from_moves, legal_actions, next_state,
                          is_terminal, get_outcome, utility,
                          render, render_svg, serialize, deserialize, to_pgn,
                          reset_transition_count, transition_count)

__all__ = [
    "ClaimDraw", "Action", "Result", "Outcome", "ChessState",
    "initial_state", "from_moves", "legal_actions", "next_state",
    "is_terminal", "get_outcome", "utility",
    "render", "render_svg", "serialize", "deserialize", "to_pgn",
    "reset_transition_count", "transition_count",
]
