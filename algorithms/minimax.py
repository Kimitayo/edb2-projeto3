"""Minimax com poda Alfa-Beta e aprofundamento iterativo.

O agente retorna a melhor ação da última profundidade concluída.
Se a profundidade 1 não terminar, retorna a primeira ação legal em ordem UCI.
"""

from __future__ import annotations

import time

import chess

from algorithms.base import Agent, Budget, Decision
from game.chess_rules import is_terminal, legal_actions, next_state, utility
from game.evaluation import evaluate

INF = float("inf")


class _BudgetExhausted(Exception):
    """Sinaliza que o orçamento acabou no meio de uma busca."""


def action_sort_key(state: chess.Board, action) -> tuple[int, str]:
    """Ordena por capturas, promoções, xeques e demais lances, com desempate UCI."""
    if not isinstance(action, chess.Move):
        return (4, "~claim_draw")

    if state.is_capture(action):
        category = 0
    elif action.promotion is not None:
        category = 1
    elif state.gives_check(action):
        category = 2
    else:
        category = 3

    return (category, action.uci())


def _uci_key(action) -> str:
    return action.uci() if isinstance(action, chess.Move) else "~claim_draw"


class MinimaxAgent(Agent):
    """Minimax com aprofundamento iterativo e poda Alfa-Beta opcional."""

    name = "minimax"

    def __init__(self, use_pruning: bool = True):
        self.use_pruning = use_pruning

    def choose_action(self, state: chess.Board, budget: Budget) -> Decision:
        if (
            budget.max_time_s is None
            and budget.max_transitions is None
            and budget.max_depth is None
        ):
            raise ValueError("Informe max_time_s, max_transitions ou max_depth no Budget.")

        t0 = time.perf_counter_ns()
        cpu0 = time.process_time_ns()

        self._deadline_ns = (
            t0 + int(budget.max_time_s * 1_000_000_000)
            if budget.max_time_s is not None
            else None
        )
        self._max_transitions = budget.max_transitions
        max_depth = budget.max_depth
        self._player = state.turn
        self._visited = 0
        self._expanded = 0
        self._transitions = 0
        self._max_ply = 0
        self._prunes = 0
        self._depth_limited = False

        root_actions = sorted(
            legal_actions(state),
            key=lambda action: action_sort_key(state, action),
        )
        if not root_actions:
            raise ValueError("Posição sem ações legais.")

        best_action = min(root_actions, key=_uci_key)
        best_value: float | None = None
        complete_depth = 0
        depth = 1

        try:
            while max_depth is None or depth <= max_depth:
                self._depth_limited = False
                value, action = self._search_root(state, root_actions, depth)

                # Só adota o resultado depois de concluir a profundidade inteira.
                best_action, best_value, complete_depth = action, value, depth

                if abs(value) == 1.0:
                    break
                if not self._depth_limited:
                    break
                depth += 1

        except _BudgetExhausted:
            pass

        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        cpu_ms = (time.process_time_ns() - cpu0) / 1e6
        overrun = (
            budget.max_time_s is not None
            and elapsed_ms > budget.max_time_s * 1000
        )

        extra = {
            "algorithm": "minimax",
            "transitions": self._transitions,
            "alpha_beta_prunes": self._prunes,
            "complete_depth": complete_depth,
            "root_value": best_value,
            "elapsed_ms": elapsed_ms,
            "cpu_ms": cpu_ms,
            "budget_overrun": overrun,
            "status": "ok",
        }

        return Decision(
            transitions=self._transitions,
            action=best_action,
            nodes_visited=self._visited,
            nodes_expanded=self._expanded,
            max_depth_reached=self._max_ply,
            extra=extra,
        )

    def _check_time(self) -> None:
        if self._deadline_ns is not None and time.perf_counter_ns() >= self._deadline_ns:
            raise _BudgetExhausted

    def _apply(self, state: chess.Board, action):
        if self._max_transitions is not None and self._transitions >= self._max_transitions:
            raise _BudgetExhausted
        self._transitions += 1
        return next_state(state, action)

    def _search_root(self, state, actions, depth: int):
        alpha, beta = -INF, INF
        best_value, best_action = -INF, actions[0]
        self._visited += 1

        for indice, action in enumerate(actions):
            self._check_time()
            child = self._apply(state, action)
            if indice == 0:
                self._expanded += 1

            value = self._search(child, depth - 1, alpha, beta, 1)
            if value > best_value:
                best_value, best_action = value, action
            if self.use_pruning and best_value > alpha:
                alpha = best_value

        return best_value, best_action

    def _search(self, state, depth: int, alpha: float, beta: float, ply: int) -> float:
        self._check_time()
        self._visited += 1
        if ply > self._max_ply:
            self._max_ply = ply

        if is_terminal(state):
            return utility(state, self._player)
        if depth == 0:
            self._depth_limited = True
            return evaluate(state, self._player)

        actions = sorted(
            legal_actions(state),
            key=lambda action: action_sort_key(state, action),
        )
        if not actions:
            return evaluate(state, self._player)

        if state.turn == self._player:  # MAX
            value = -INF
            for indice, action in enumerate(actions):
                child = self._apply(state, action)
                if indice == 0:
                    self._expanded += 1

                value = max(value, self._search(child, depth - 1, alpha, beta, ply + 1))
                if self.use_pruning:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        self._prunes += 1
                        break
            return value

        value = INF  # MIN
        for indice, action in enumerate(actions):
            child = self._apply(state, action)
            if indice == 0:
                self._expanded += 1

            value = min(value, self._search(child, depth - 1, alpha, beta, ply + 1))
            if self.use_pruning:
                beta = min(beta, value)
                if alpha >= beta:
                    self._prunes += 1
                    break
        return value