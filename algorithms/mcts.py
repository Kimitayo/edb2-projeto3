"""MCTS clássico (UCT) e utilitários compartilhados com o GPN-MCTS.

Responsável: Gustavo (Integrante 3).
Depende de: game (regras, Integrante 1), algorithms/base.py e, opcionalmente,
game/evaluation.py (Flávio). Enquanto `evaluate` não existir, usa-se um
fallback provisório baseado em saldo de material.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import chess

from game import Action, is_terminal, legal_actions, next_state, transition_count, utility
from algorithms.base import Agent, Budget, Decision

PLAYOUT_LIMIT = 80            # meios-lances por playout (plano do projeto)
DEFAULT_C = math.sqrt(2)      # constante de exploração UCT
_EPS = 1e-12
OVERRUN_TOLERANCE_MS = 1.0    # folga de `budget_overrun`: a busca para ao cruzar o prazo

# --- Heurística (H) -----------------------------------------------------------
try:
    from game.evaluation import evaluate as _evaluate  # Integrante 2 (Flávio)
except ImportError:  # módulo ainda não existe: usa o fallback provisório
    _evaluate = None

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}
_MAX_MATERIAL = 39  # 8 peões + 2 cavalos(3) + 2 bispos(3) + 2 torres(5) + dama(9)


def _provisional_eval_white(state: chess.Board) -> float:
    """Fallback provisório: saldo de material normalizado para [-1, 1]."""
    balance = 0
    for piece_type, value in _PIECE_VALUES.items():
        balance += value * (len(state.pieces(piece_type, chess.WHITE))
                            - len(state.pieces(piece_type, chess.BLACK)))
    return balance / _MAX_MATERIAL


def heuristic_white(state: chess.Board) -> float:
    """Heurística H em [-1, 1], do ponto de vista das brancas (não terminal)."""
    if _evaluate is not None:
        value = float(_evaluate(state, chess.WHITE))
    else:
        value = _provisional_eval_white(state)
    return max(-1.0, min(1.0, value))


def value_white(state: chess.Board) -> float:
    """Utilidade U se terminal; heurística H caso contrário (ótica das brancas)."""
    if is_terminal(state):
        return utility(state, chess.WHITE)
    return heuristic_white(state)


# --- Orçamento ----------------------------------------------------------------
class BudgetExhausted(Exception):
    """Sinaliza que o orçamento acabou no meio de uma iteração."""


class Clock:
    """Controla o orçamento. Métrica principal: transições (next_state).
    Tempo é opcional e só deve ser usado em experimentos de tempo real."""

    def __init__(self, budget: Budget) -> None:
        if budget.max_transitions is None and budget.max_time_ms is None:
            raise ValueError("Orçamento sem limite: defina max_transitions ou max_time_ms.")
        self.max_transitions = budget.max_transitions
        self.max_time_ms = budget.max_time_ms
        self._t0 = time.perf_counter_ns()      # tempo real (Seção 8.3)
        self._cpu0 = time.process_time_ns()    # tempo de CPU (Seção 8.3)
        self._deadline = (None if budget.max_time_ms is None
                          else self._t0 + int(budget.max_time_ms * 1_000_000))
        self._tr0 = transition_count()

    @property
    def transitions(self) -> int:
        return transition_count() - self._tr0

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self._t0) / 1e6

    def cpu_ms(self) -> float:
        return (time.process_time_ns() - self._cpu0) / 1e6

    def exhausted(self) -> bool:
        if self.max_transitions is not None and self.transitions >= self.max_transitions:
            return True
        if self._deadline is not None and time.perf_counter_ns() >= self._deadline:
            return True
        return False

    def overrun(self) -> bool:
        """Orçamento ultrapassado (métrica `budget_overrun`, Seção 8.4)."""
        if self.max_transitions is not None and self.transitions > self.max_transitions:
            return True
        return (self._deadline is not None and time.perf_counter_ns()
                > self._deadline + int(OVERRUN_TOLERANCE_MS * 1_000_000))

    def common_extra(self) -> dict:
        """Métricas de custo comuns (Seção 8.4), para o `extra` das decisões."""
        return {"elapsed_ms": self.elapsed_ms(), "cpu_ms": self.cpu_ms(),
                "budget_overrun": self.overrun()}

    def check(self) -> None:
        if self.exhausted():
            raise BudgetExhausted()


# --- Playout ------------------------------------------------------------------
def playout(state: chess.Board, rng: random.Random, clock: Clock,
            limit: int = PLAYOUT_LIMIT) -> tuple[float, int]:
    """Playout com lances legais pseudoaleatórios.
    Retorna (valor das brancas, passos). Se o limite for atingido, o valor é H,
    que nunca é convertido em prova."""
    current = state
    steps = 0
    while not is_terminal(current) and steps < limit:
        clock.check()
        current = next_state(current, rng.choice(legal_actions(current)))
        steps += 1
    return value_white(current), steps


# --- Árvore -------------------------------------------------------------------
class Node:
    """Nó da árvore. `total` acumula utilidades do ponto de vista de `mover`,
    o jogador que escolheu o lance que levou a este nó (None na raiz)."""

    __slots__ = ("state", "parent", "action", "mover", "depth", "children",
                 "untried", "terminal", "visits", "total", "exhausted")

    def __init__(self, state: chess.Board, parent: Optional["Node"] = None,
                 action: Optional[Action] = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.mover = parent.state.turn if parent is not None else None
        self.depth = 0 if parent is None else parent.depth + 1
        self.children: dict[Action, "Node"] = {}
        self.terminal = is_terminal(state)
        self.untried: list[Action] = [] if self.terminal else list(legal_actions(state))
        self.visits = 0
        self.total = 0.0
        # Subárvore totalmente resolvida (sem trabalho a fazer). Usado só para parar.
        self.exhausted = self.terminal


@dataclass
class SearchStats:
    iterations: int = 0
    playouts: int = 0
    playout_steps: int = 0
    nodes_expanded: int = 0   # nós criados na árvore (um por expansão)
    nodes_visited: int = 0    # nós percorridos na seleção/expansão
    max_depth: int = 0


def _select_child(node: Node, rng: random.Random, c: float) -> Node:
    """UCB1: maximiza v_i + C * sqrt(ln N_p / n_i). Empates: sorteio por UCI."""
    log_n = math.log(node.visits)
    best: list[Node] = []
    best_score = -math.inf
    for child in node.children.values():
        score = child.total / child.visits + c * math.sqrt(log_n / child.visits)
        if score > best_score + _EPS:
            best, best_score = [child], score
        elif score >= best_score - _EPS:
            best.append(child)
    if len(best) == 1:
        return best[0]
    return rng.choice(sorted(best, key=lambda ch: ch.action.uci()))


# --- Agente -------------------------------------------------------------------
class MCTSAgent(Agent):
    """MCTS clássico (UCT) com playout pseudoaleatório limitado."""
    name = "mcts"

    def __init__(self, seed: int = 0, c: float = DEFAULT_C,
                 playout_limit: int = PLAYOUT_LIMIT) -> None:
        self.seed = seed
        self.c = c
        self.playout_limit = playout_limit

    def choose_action(self, state: chess.Board, budget: Budget) -> Decision:
        if is_terminal(state):
            raise ValueError("Não há ação a escolher em estado terminal.")

        clock = Clock(budget)
        rng = random.Random(self.seed)  # RNG local: decisões reprodutíveis
        root = Node(state)
        stats = SearchStats()

        try:
            while not root.exhausted and not clock.exhausted():
                self._iterate(root, rng, clock, stats)
        except BudgetExhausted:
            pass  # a iteração interrompida não é retropropagada

        kids = [ch for ch in root.children.values() if ch.visits > 0]
        if kids:  # escolha final: mais visitas; empate por UCI
            best = min(kids, key=lambda ch: (-ch.visits, ch.action.uci()))
            action, fallback = best.action, False
        else:     # orçamento acabou antes de qualquer iteração completa
            action = min(legal_actions(state), key=lambda a: a.uci())
            fallback = True

        return Decision(
            action=action,
            nodes_visited=stats.nodes_visited,
            nodes_expanded=stats.nodes_expanded,
            transitions=clock.transitions,
            max_depth_reached=stats.max_depth,
            extra={
                "algorithm": "mcts",
                "iterations": stats.iterations,
                "playouts": stats.playouts,
                "playout_steps": stats.playout_steps,
                "tree_nodes_created": 1 + stats.nodes_expanded,
                "root_visits": root.visits,
                **clock.common_extra(),
                "fallback": fallback,
            },
        )

    def _iterate(self, root: Node, rng: random.Random, clock: Clock,
                 stats: SearchStats) -> None:
        """Uma iteração: seleção, expansão, playout e retropropagação."""
        node = root
        path = [root]
        stats.nodes_visited += 1

        # 1) Seleção
        while not node.terminal and not node.untried:
            node = _select_child(node, rng, self.c)
            path.append(node)
            stats.nodes_visited += 1

        # 2) Expansão (um filho ainda não criado, sorteado uniformemente)
        if node.untried:
            clock.check()
            action = node.untried.pop(rng.randrange(len(node.untried)))
            child = Node(next_state(node.state, action), node, action)
            node.children[action] = child
            node = child
            path.append(child)
            stats.nodes_expanded += 1
            stats.nodes_visited += 1
            stats.max_depth = max(stats.max_depth, child.depth)

        # 3) Playout
        value, steps = playout(node.state, rng, clock, self.playout_limit)
        stats.playouts += 1
        stats.playout_steps += steps

        # 4) Retropropagação
        for n in reversed(path):
            n.visits += 1
            if n.mover is not None:
                n.total += value if n.mover == chess.WHITE else -value
            n.exhausted = n.terminal or (
                not n.untried and all(ch.exhausted for ch in n.children.values())
            )

        stats.iterations += 1
