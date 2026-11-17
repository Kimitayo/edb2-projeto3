"""
results_storage.py - Tabelas CSV dos experimentos (Plano v2.0, secoes 8.4, 9.4 e 9.5).

Tres tabelas, sempre com o mesmo cabecalho:
  decisions.csv : UMA linha = UMA decisao de UM agente (posicao ou partida)
  quality.csv   : UMA linha = qualidade (Stockfish) de UMA decisao de decisions.csv
  games.csv     : UMA linha = UMA partida completa

Regras:
  - cada linha e gravada e "flushada" na hora (se travar, nada se perde);
  - arquivos brutos NUNCA sao editados a mao (plano, secao 12.4);
  - valores ausentes ficam vazios (None).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path

STATUSES = ("ok", "timeout", "illegal_action", "error")
RUN_KINDS = ("time", "memory", "game")      # rodada de tempo, de memoria ou partida
BUDGET_TYPES = ("time", "transitions")
AGENTS = ("minimax", "gpn_mcts", "hybrid")
POSITION_SETS = ("calibration", "test", "game")


@dataclass
class DecisionRecord:
    # --- identificacao ---
    experiment_id: str
    run_id: int
    run_kind: str               # "time", "memory" ou "game"
    position_id: str            # ex.: "cal_C3_07" (vazio nas partidas)
    position_set: str           # "calibration", "test" ou "game"
    context: str                # "C1".."C6" (vazio nas partidas)
    game_id: str                # vazio fora das partidas
    ply: int | None             # meio-lance da partida (vazio fora das partidas)
    agent: str
    seed: int | None            # 2026/2027/2028 para GPN-MCTS e hibrida; vazio no Minimax
    repetition: int
    budget_type: str            # "time" ou "transitions"
    budget_value: float         # ms ou numero de transicoes
    git_commit: str

    # --- acao ---
    action_uci: str             # "e2e4" ou "CLAIM_DRAW"
    action_changed: bool | None  # mudou de acao em relacao a repeticao 0 (mesma seed)?

    # --- metricas comuns (secao 8.4) ---
    nodes_visited: int | None
    nodes_expanded: int | None
    tree_nodes_created: int | None
    transitions: int | None
    max_depth_reached: int | None
    elapsed_ms: float
    cpu_ms: float
    peak_memory_kb: float | None   # so nas linhas run_kind == "memory"
    budget_overrun: bool
    status: str

    # --- especificas do Minimax ---
    alpha_beta_prunes: int | None = None
    complete_depth: int | None = None
    # --- especificas do GPN-MCTS ---
    iterations: int | None = None
    playouts: int | None = None
    playout_steps: int | None = None
    proof_updates: int | None = None
    pnrank_sorts: int | None = None
    # --- especificas da hibrida ---
    chosen_algorithm: str = ""
    hybrid_context: str = ""
    selection_ms: float | None = None
    p_minimax: float | None = None
    p_gpn_mcts: float | None = None

    # --- controle ---
    error_message: str = ""
    extra_json: str = ""        # o que sobrar de `Decision.extra`
    timestamp: str = ""


@dataclass
class QualityRecord:
    experiment_id: str
    run_id: int                 # liga com decisions.csv
    position_id: str
    agent: str
    action_uci: str
    ref_best_uci: str
    top1: bool                  # jogada == melhor jogada do Stockfish
    ref_best_cp: int | None     # do ponto de vista de quem joga; vazio se mate
    ref_best_mate: int | None
    agent_cp: int | None
    agent_mate: int | None
    regret_cp: float | None     # max(0, melhor - agente); vazio se envolve mate
    decisive_class: str         # "cp", "mate_kept", "mate_missed", ...
    ref_engine: str
    ref_depth: int
    timestamp: str = ""


@dataclass
class GameRecord:
    experiment_id: str
    game_id: str
    matchup: str                # ex.: "minimax_vs_gpn_mcts"
    white: str
    black: str
    seed_white: int | None
    seed_black: int | None
    budget_type: str
    budget_value: float
    result: str                 # "1-0", "0-1", "1/2-1/2" ou "*" (falha/incompleta)
    status: str                 # "completed", "failure" ou "max_plies"
    termination: str            # ex.: "checkmate", "stalemate", "claim_draw", "illegal_action"
    plies: int
    failures: int               # decisoes com status diferente de "ok"
    points_white: float | None  # vitoria 1, empate 0,5, derrota 0
    points_black: float | None
    pgn_path: str
    git_commit: str
    timestamp: str = ""


def _validate_decision(r: DecisionRecord) -> None:
    if r.status not in STATUSES:
        raise ValueError(f"status invalido: {r.status!r}")
    if r.run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind invalido: {r.run_kind!r}")
    if r.budget_type not in BUDGET_TYPES:
        raise ValueError(f"budget_type invalido: {r.budget_type!r}")
    if r.agent not in AGENTS:
        raise ValueError(f"agente desconhecido: {r.agent!r}")
    if r.position_set not in POSITION_SETS:
        raise ValueError(f"position_set invalido: {r.position_set!r}")
    if (r.nodes_visited is not None and r.nodes_expanded is not None
            and r.nodes_expanded > r.nodes_visited):
        raise ValueError("nodes_expanded nao pode ser maior que nodes_visited")
    if r.run_kind != "memory" and r.peak_memory_kb is not None:
        raise ValueError("peak_memory_kb so pode ser preenchido na rodada de memoria")


def _validate_game(r: GameRecord) -> None:
    if r.result not in ("1-0", "0-1", "1/2-1/2", "*"):
        raise ValueError(f"result invalido: {r.result!r}")
    if r.status not in ("completed", "failure", "max_plies"):
        raise ValueError(f"status da partida invalido: {r.status!r}")
    if r.status != "completed" and r.result != "*":
        raise ValueError("falha ou partida incompleta nunca vira empate: result deve ser '*'")


class CsvTable:
    """Abre (ou cria) um CSV e adiciona linhas validadas."""

    def __init__(self, path: str | Path, record_cls, validator=None):
        self.path = Path(path)
        self.record_cls = record_cls
        self.validator = validator
        self.columns = [f.name for f in fields(record_cls)]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.columns)
        if new_file:
            self._w.writeheader()
            self._f.flush()

    def write(self, record) -> None:
        if not isinstance(record, self.record_cls):
            raise TypeError(f"esperado {self.record_cls.__name__}")
        if hasattr(record, "timestamp") and not record.timestamp:
            record.timestamp = datetime.now().isoformat(timespec="seconds")
        if self.validator:
            self.validator(record)
        self._w.writerow(asdict(record))
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def decision_table(path) -> CsvTable:
    return CsvTable(path, DecisionRecord, _validate_decision)


def quality_table(path) -> CsvTable:
    return CsvTable(path, QualityRecord)


def game_table(path) -> CsvTable:
    return CsvTable(path, GameRecord, _validate_game)


def load_rows(path: str | Path) -> list[dict]:
    """Le um CSV de volta (valores como texto)."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def next_run_id(path: str | Path) -> int:
    """Proximo run_id livre (para continuar um CSV que ja existe)."""
    ids = [int(r["run_id"]) for r in load_rows(path) if r.get("run_id")]
    return max(ids) + 1 if ids else 1
