"""
results_storage.py - Guarda os resultados dos experimentos em CSV.

Regras para os dados serem confiaveis:
  1. UMA linha = UMA batalha de UM algoritmo em UM cenario (uma repeticao).
  2. O cabecalho e sempre o mesmo (vem de RunRecord) - nada de colunas soltas.
  3. Cada linha e gravada e "flushada" na hora: se o programa travar,
     nao perde o que ja rodou.
  4. Arquivos de resultado brutos NUNCA sao editados a mao.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path


@dataclass
class RunRecord:
    # --- identificacao (para reproduzir) ---
    experiment_id: str          # ex.: "exp01_orcamento_nos"
    run_id: int                 # numero sequencial da linha
    scenario_id: str            # ex.: "sc_0042"
    scenario_seed: int          # semente que gerou o cenario
    run_seed: int               # semente da aleatoriedade do algoritmo/jogo
    repetition: int             # 0..R-1
    git_commit: str             # commit do codigo usado

    # --- configuracao ---
    algorithm: str              # ex.: "alphabeta", "mcts"
    opponent: str               # algoritmo adversario
    budget_type: str            # "nodes" ou "time"
    budget_value: float         # ex.: 1000 (nos) ou 0.05 (s)
    scenario_difficulty: str    # ex.: "small" / "medium" / "large"

    # --- resultados de qualidade ---
    winner: str                 # "A", "B" ou "draw"
    turns: int                  # numero de turnos da batalha
    hp_remaining_a: float
    hp_remaining_b: float

    # --- resultados de custo ---
    decisions: int
    nodes_visited: int
    nodes_expanded: int
    max_depth: int
    total_time_ms: float
    mean_decision_time_ms: float
    peak_memory_kb: float | None  # vazio quando a memoria nao foi medida
    budget_exhausted_count: int

    # --- controle ---
    status: str = "ok"          # "ok", "error" ou "timeout"
    timestamp: str = ""         # preenchido automaticamente


COLUMNS = [f.name for f in fields(RunRecord)]


class ResultsWriter:
    """Abre (ou cria) um CSV e adiciona linhas validadas."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS)
        if new_file:
            self._w.writeheader()
            self._f.flush()

    def write(self, record: RunRecord) -> None:
        if not record.timestamp:
            record.timestamp = datetime.now().isoformat(timespec="seconds")
        self._validate(record)
        self._w.writerow(asdict(record))
        self._f.flush()

    @staticmethod
    def _validate(r: RunRecord) -> None:
        if r.nodes_expanded > r.nodes_visited:
            raise ValueError("nodes_expanded nao pode ser maior que nodes_visited")
        if r.winner not in ("A", "B", "draw"):
            raise ValueError(f"winner invalido: {r.winner!r}")
        if r.budget_type not in ("nodes", "time"):
            raise ValueError(f"budget_type invalido: {r.budget_type!r}")
        if r.status not in ("ok", "error", "timeout"):
            raise ValueError(f"status invalido: {r.status!r}")

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_results(path: str | Path) -> list[dict]:
    """Le o CSV de volta (valores como texto; converta ao analisar)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
