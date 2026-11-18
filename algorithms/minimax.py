"""Minimax com poda Alfa-Beta e aprofundamento iterativo (Seção 6 da especificação).

Contrato: MinimaxAgent.choose_action(state, budget) -> Decision

Decisões fixas da especificação:
- alternância MAX/MIN do ponto de vista do jogador da raiz;
- utilidade terminal U (+1/0/-1) e heurística H apenas em folhas não terminais;
- sem tabela de transposição;
- ordenação: capturas, promoções, xeques, demais; desempate pela notação UCI;
- retorna a melhor ação da última profundidade COMPLETAMENTE concluída;
- se nem a profundidade 1 terminar, retorna a primeira ação legal em ordem UCI.
"""

from __future__ import annotations

import time
from dataclasses import fields

import chess

# Ajuste os imports abaixo se os módulos de vocês tiverem nomes diferentes.
from algorithms.base import Agent, Budget, Decision
from game.chess_rules import is_terminal, legal_actions, next_state, utility
from game.evaluation import evaluate

INF = float("inf")


class _BudgetExhausted(Exception):
    """Sinaliza que o orçamento acabou no meio de uma busca."""


def action_sort_key(state: chess.Board, action) -> tuple[int, str]:
    """Ordenação determinística: capturas, promoções, xeques, demais; desempate UCI."""
    if not isinstance(action, chess.Move):  # CLAIM_DRAW (ação especial)
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
    """Minimax + Alfa-Beta com aprofundamento iterativo.

    use_pruning=False desativa a poda (serve para comparar com Alfa-Beta em
    árvores pequenas). A profundidade máxima vem de Budget.max_depth
    (obrigatória se o Budget não tiver tempo nem transições).
    """

    name = "minimax"

    def __init__(self, use_pruning: bool = True):
        self.use_pruning = use_pruning

    # ------------------------------------------------------------------ API
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
            t0 + int(budget.max_time_s * 1_000_000_000) if budget.max_time_s is not None else None
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

        root_actions = sorted(legal_actions(state), key=lambda a: action_sort_key(state, a))
        if not root_actions:
            raise ValueError("Posição sem ações legais.")

        # Fallback: primeira ação legal em ordem UCI.
        best_action = min(root_actions, key=_uci_key)
        best_value: float | None = None
        complete_depth = 0

        depth = 1
        try:
            while max_depth is None or depth <= max_depth:
                self._depth_limited = False
                value, action = self._search_root(state, root_actions, depth)
                # Profundidade concluída por inteiro: pode ser adotada.
                best_action, best_value, complete_depth = action, value, depth

                if abs(value) == 1.0:  # vitória/derrota comprovada (H nunca chega a ±1)
                    break
                if not self._depth_limited:  # árvore inteira explorada
                    break
                # Próxima iteração parte da melhor ação primeiro (acelera a poda);
                # o restante mantém a ordenação determinística.
                root_actions.remove(action)
                root_actions.insert(0, action)
                depth += 1
        except _BudgetExhausted:
            pass

        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        cpu_ms = (time.process_time_ns() - cpu0) / 1e6
        overrun = budget.max_time_s is not None and elapsed_ms > budget.max_time_s * 1000

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
        kwargs = dict(
            action=best_action,
            nodes_visited=self._visited,
            nodes_expanded=self._expanded,
            max_depth_reached=self._max_ply,
            extra=extra,
        )
        # Contrato da spec (Seção 12.2) tem o campo `transitions`; o base.py atual
        # ainda não. Passa o campo só se ele existir.
        if "transitions" in {f.name for f in fields(Decision)}:
            kwargs["transitions"] = self._transitions
        return Decision(**kwargs)

    # ------------------------------------------------------------- internos
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
        self._expanded += 1
        for action in actions:
            self._check_time()
            child = self._apply(state, action)
            value = self._search(child, depth - 1, alpha, beta, 1)
            if value > best_value:  # estrito: empates ficam com a primeira na ordenação
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

        actions = sorted(legal_actions(state), key=lambda a: action_sort_key(state, a))
        self._expanded += 1
        if not actions:  # defensivo: sem ações e não terminal
            return evaluate(state, self._player)

        if state.turn == self._player:  # MAX
            value = -INF
            for action in actions:
                child = self._apply(state, action)
                value = max(value, self._search(child, depth - 1, alpha, beta, ply + 1))
                if self.use_pruning:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        self._prunes += 1
                        break
            return value

        value = INF  # MIN
        for action in actions:
            child = self._apply(state, action)
            value = min(value, self._search(child, depth - 1, alpha, beta, ply + 1))
            if self.use_pruning:
                beta = min(beta, value)
                if alpha >= beta:
                    self._prunes += 1
                    break
        return value
