
"""Interface padrão dos agentes (Minimax, GPN-MCTS, Híbrido e baselines).

Contrato comum:
    choose_action(state, budget) -> Decision

O orçamento pode ser definido por:
    - max_time_ms: tempo em milissegundos (contrato oficial).
    - max_transitions: quantidade de transições.

Compatibilidade:
    - max_time_s: formato antigo, em segundos.
    - max_depth: limite adicional utilizado pelo Minimax.

A Decision registra as métricas comuns de todos os algoritmos.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import chess

from game import Action, legal_actions



@dataclass(frozen=True)
class Budget:
    """Orçamento computacional de uma decisão."""

    max_transitions: Optional[int] = None
    max_time_s: Optional[float] = None
    max_depth: Optional[int] = None
    max_time_ms: Optional[float] = None

    def __post_init__(self) -> None:
        """Sincroniza os formatos de tempo."""

        if self.max_time_ms is not None and self.max_time_s is not None:
            expected_ms = self.max_time_s * 1000.0

            if not math.isclose(
                float(self.max_time_ms),
                float(expected_ms),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "max_time_ms e max_time_s representam "
                    "tempos diferentes."
                )

        elif self.max_time_s is not None:
            object.__setattr__(
                self,
                "max_time_ms",
                self.max_time_s * 1000.0,
            )

        elif self.max_time_ms is not None:
            object.__setattr__(
                self,
                "max_time_s",
                self.max_time_ms / 1000.0,
            )


def __post_init__(self) -> None:
    """Sincroniza os formatos de tempo."""

    # Se os dois formatos forem informados, verifica
    # se representam o mesmo tempo.
    if self.max_time_ms is not None and self.max_time_s is not None:
        expected_ms = self.max_time_s * 1000.0

        if not math.isclose(
            float(self.max_time_ms),
            float(expected_ms),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "max_time_ms e max_time_s representam "
                "tempos diferentes."
            )

    # Segundos -> milissegundos.
    elif self.max_time_s is not None:
        object.__setattr__(
            self,
            "max_time_ms",
            self.max_time_s * 1000.0,
        )

    # Milissegundos -> segundos.
    elif self.max_time_ms is not None:
        object.__setattr__(
            self,
            "max_time_s",
            self.max_time_ms / 1000.0,
        )


@dataclass
class Decision:
    """Resultado de uma decisão tomada por um agente.

    transitions permanece opcional durante a migração dos agentes antigos.
    Quando None, o runner pode utilizar o contador do motor de xadrez.
    """

    action: Action

    nodes_expanded: int = 0
    nodes_visited: int = 0
    max_depth_reached: int = 0

    extra: dict = field(default_factory=dict)

    # Campo do contrato oficial.
    # Mantido após os campos antigos para preservar compatibilidade
    # com chamadas posicionais existentes.
    transitions: Optional[int] = None


class Agent(ABC):
    """Interface abstrata dos agentes de busca."""

    name: str = "agent"

    @abstractmethod
    def choose_action(
        self,
        state: chess.Board,
        budget: Budget,
    ) -> Decision:
        """Escolhe uma ação para o estado recebido."""
        ...


class RandomAgent(Agent):
    """Baseline: escolhe uma ação legal aleatoriamente."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose_action(self, state, budget):
        return Decision(
            action=self.rng.choice(legal_actions(state)),
            transitions=0,
        )
