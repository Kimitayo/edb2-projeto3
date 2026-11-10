"""Solver EXATO por força bruta com memoização (negamax sem poda).

Serve apenas como GABARITO para validar o simulador e as decisões dos
algoritmos em cenários pequenos. Não é a implementação de Minimax do
projeto (essa é do Integrante 2, com Alfa-Beta e métricas).
"""
from __future__ import annotations

from functools import lru_cache

from game import BattleState, is_terminal, legal_actions, next_state, utility


@lru_cache(maxsize=None)
def solve(state: BattleState) -> float:
    """Valor do jogo (+1/0/-1) para o jogador da vez, com jogo perfeito."""
    if is_terminal(state):
        return utility(state, state.to_move)
    return max(-solve(next_state(state, a)) for a in legal_actions(state))


def optimal_actions(state: BattleState) -> set[int]:
    """Conjunto de ações que preservam o valor ótimo."""
    v = solve(state)
    return {a for a in legal_actions(state) if -solve(next_state(state, a)) == v}


if __name__ == "__main__":  # imprime o gabarito de todos os cenários
    import sys
    sys.setrecursionlimit(10_000)
    from game.scenarios import SCENARIOS
    for name, make in SCENARIOS.items():
        s = make()
        solve.cache_clear()
        v = solve(s)
        print(f"{name:16} valor={v:+.0f} ótimas={sorted(optimal_actions(s))} "
              f"legais={list(legal_actions(s))} estados={solve.cache_info().currsize}")
