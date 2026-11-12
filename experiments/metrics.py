"""
metrics.py - Modulo inicial de metricas dos experimentos.

Ideia central: os algoritmos NAO medem nada sozinhos. Eles recebem um
`SearchCounter` e chamam counter.visit() / counter.expand(). Assim a contagem
e feita do MESMO jeito para todos os algoritmos (comparacao justa).

Uso tipico (dentro de um algoritmo):

    def choose_action(state, counter):
        counter.visit(depth=0)
        ...
        counter.expand()          # quando gerar os filhos de um no
        if counter.exhausted():   # orcamento acabou -> devolve a melhor jogada ate agora
            return best_so_far
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Levantada por counter.check() quando o orcamento acaba."""


@dataclass(frozen=True)
class Budget:
    """Orcamento computacional POR DECISAO (por jogada).

    max_nodes   : limite de nos visitados (orcamento principal, justo e reproduzivel)
    max_time_s  : limite de tempo em segundos (orcamento secundario)
    Se um dos dois for None, ele nao e usado.
    """
    max_nodes: int | None = None
    max_time_s: float | None = None

    def label(self) -> str:
        if self.max_nodes is not None and self.max_time_s is None:
            return f"nodes={self.max_nodes}"
        if self.max_time_s is not None and self.max_nodes is None:
            return f"time={self.max_time_s}s"
        return f"nodes={self.max_nodes};time={self.max_time_s}s"


class SearchCounter:
    """Conta nos visitados/expandidos de UMA decisao e vigia o orcamento.

    Definicoes (devem ser as mesmas no artigo inteiro):
      - no VISITADO : todo estado em que o algoritmo "entra" para avaliar
                      (inclui folhas e estados avaliados por heuristica/simulacao).
      - no EXPANDIDO: todo estado para o qual o algoritmo GEROU os sucessores.
    Portanto: expandidos <= visitados.
    """

    def __init__(self, budget: Budget | None = None):
        self.budget = budget or Budget()
        self.visited = 0
        self.expanded = 0
        self.max_depth = 0
        self._t0 = time.perf_counter()

    def visit(self, depth: int = 0) -> None:
        self.visited += 1
        if depth > self.max_depth:
            self.max_depth = depth

    def expand(self) -> None:
        self.expanded += 1

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0

    def exhausted(self) -> bool:
        b = self.budget
        if b.max_nodes is not None and self.visited >= b.max_nodes:
            return True
        if b.max_time_s is not None and self.elapsed_s() >= b.max_time_s:
            return True
        return False

    def check(self) -> None:
        """Alternativa a exhausted(): levanta excecao se o orcamento acabou."""
        if self.exhausted():
            raise BudgetExceeded()


@dataclass
class Measurement:
    """Resultado de medir UMA decisao."""
    wall_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    peak_memory_kb: float | None = None  # None quando a memoria nao foi medida


@contextmanager
def measure(track_memory: bool = False):
    """Mede tempo (e opcionalmente memoria) do bloco.

    IMPORTANTE: tracemalloc deixa o Python mais lento. Por isso o protocolo
    usa DUAS rodadas: uma com track_memory=False (para TEMPO) e outra com
    track_memory=True (para MEMORIA). Nunca use o tempo da rodada de memoria.

        with measure(track_memory=False) as m:
            action = algoritmo.choose_action(state, counter)
        print(m.wall_time_ms)
    """
    m = Measurement()
    started_here = False
    if track_memory:
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
            started_here = True
    t_wall = time.perf_counter_ns()
    t_cpu = time.process_time_ns()
    try:
        yield m
    finally:
        m.wall_time_ms = (time.perf_counter_ns() - t_wall) / 1e6
        m.cpu_time_ms = (time.process_time_ns() - t_cpu) / 1e6
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            m.peak_memory_kb = peak / 1024
            if started_here:
                tracemalloc.stop()


@dataclass
class BattleAggregator:
    """Acumula as metricas de TODAS as decisoes de uma batalha (uma partida)."""
    decisions: int = 0
    nodes_visited: int = 0
    nodes_expanded: int = 0
    max_depth: int = 0
    total_time_ms: float = 0.0
    peak_memory_kb: float | None = None
    budget_exhausted_count: int = 0

    def add(self, counter: SearchCounter, m: Measurement) -> None:
        self.decisions += 1
        self.nodes_visited += counter.visited
        self.nodes_expanded += counter.expanded
        self.max_depth = max(self.max_depth, counter.max_depth)
        self.total_time_ms += m.wall_time_ms
        if counter.exhausted():
            self.budget_exhausted_count += 1
        if m.peak_memory_kb is not None:
            self.peak_memory_kb = max(self.peak_memory_kb or 0.0, m.peak_memory_kb)

    @property
    def mean_decision_time_ms(self) -> float:
        return self.total_time_ms / self.decisions if self.decisions else 0.0
