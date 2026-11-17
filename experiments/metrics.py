"""
metrics.py - Medicao de tempo, CPU e memoria (Plano v2.0, secoes 8.3 e 8.4).

Os CONTADORES (nos visitados, expandidos, transicoes, profundidade) NAO ficam
aqui: eles vem dentro do `Decision` que cada agente devolve (contrato em
algorithms/base.py). Este modulo mede o que o agente nao consegue medir sozinho
e decide se o orcamento foi respeitado.

Regras do plano:
  - tempo real: time.perf_counter_ns()
  - tempo de CPU: time.process_time_ns()
  - memoria: tracemalloc, SEMPRE em execucao separada (ele deixa o codigo lento)
"""
from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass

# O plano nao define quando uma decisao vira "timeout" (so diz que tempo
# estourado e registrado). Convencao adotada: "overrun" = passou do orcamento;
# "timeout" = passou do DOBRO do orcamento. Ajustavel aqui, em um unico lugar.
TIMEOUT_FACTOR = 2.0


@dataclass
class Measurement:
    """Resultado de medir UMA decisao."""
    elapsed_ms: float = 0.0
    cpu_ms: float = 0.0
    peak_memory_kb: float | None = None  # None quando a memoria nao foi medida


@contextmanager
def measure(track_memory: bool = False):
    """Mede tempo (e, se pedido, memoria) do bloco.

        with measure() as m:
            decision = agent.choose_action(board, budget)
        print(m.elapsed_ms, m.cpu_ms)

    Com track_memory=True o tempo medido NAO vale para analise de tempo.
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
        m.elapsed_ms = (time.perf_counter_ns() - t_wall) / 1e6
        m.cpu_ms = (time.process_time_ns() - t_cpu) / 1e6
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            m.peak_memory_kb = peak / 1024
            if started_here:
                tracemalloc.stop()


def budget_parts(budget) -> tuple[str, float]:
    """('time', 300.0) ou ('transitions', 2000.0). Um tipo de limite por vez.

    Aceita os dois nomes de campo de tempo: `max_time_ms` (plano, secao 12.2) e
    `max_time_s` (algorithms/base.py atual, em segundos). O valor sai SEMPRE em ms.
    """
    max_time = getattr(budget, "max_time_ms", None)
    if max_time is None and getattr(budget, "max_time_s", None) is not None:
        max_time = budget.max_time_s * 1000.0
    max_trans = getattr(budget, "max_transitions", None)
    if (max_time is None) == (max_trans is None):
        raise ValueError("Budget deve ter exatamente UM limite: tempo ou max_transitions")
    if max_time is not None:
        return "time", float(round(max_time, 6))
    return "transitions", float(max_trans)


def budget_overrun(budget, elapsed_ms: float, transitions: int | None) -> bool:
    """True se a decisao passou do orcamento."""
    kind, value = budget_parts(budget)
    if kind == "time":
        return elapsed_ms > value
    return transitions is not None and transitions > value


def is_timeout(budget, elapsed_ms: float) -> bool:
    """True se o tempo real passou de TIMEOUT_FACTOR x o orcamento de tempo."""
    kind, value = budget_parts(budget)
    return kind == "time" and elapsed_ms > value * TIMEOUT_FACTOR
