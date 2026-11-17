"""Interface padrão dos agentes (Minimax, GPN-MCTS, Híbrido e baselines).

Todo algoritmo implementa:

    class MeuAgente(Agent):
        name = "minimax"
        def choose_action(self, state, budget) -> Decision

Orçamento (Budget): a métrica PRINCIPAL é `max_transitions` (número de
chamadas a next_state), porque é determinística e independe da máquina.
`max_time_s` é opcional e deve ser usado só em experimentos de tempo real.

Obs.: o contrato final de Budget/Decision está na Seção 12.2 de
docs/PLANO_PROJETO_XADREZ_EDBII.md (ex.: max_time_ms, campo transitions).
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import chess

from game import Action, legal_actions


@dataclass(frozen=True)
class Budget:
    max_transitions: Optional[int] = None  # chamadas a next_state
    max_time_s: Optional[float] = None
    max_depth: Optional[int] = None        # usado pelo Minimax, se quiser


@dataclass
class Decision:
    action: Action               # chess.Move ou ClaimDraw
    nodes_expanded: int = 0      # nós cujos filhos foram gerados
    nodes_visited: int = 0       # nós avaliados/visitados
    max_depth_reached: int = 0
    extra: dict = field(default_factory=dict)  # ex.: proof numbers, podas, iterações


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def choose_action(self, state: chess.Board, budget: Budget) -> Decision: ...


class RandomAgent(Agent):
    """Baseline: escolhe uma ação legal ao acaso (semente controlada)."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose_action(self, state, budget):
        return Decision(self.rng.choice(legal_actions(state)))
