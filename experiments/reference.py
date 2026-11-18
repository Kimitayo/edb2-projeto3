"""
reference.py - Stockfish como REFERENCIA de qualidade (Plano v2.0, secoes 9.3 e 9.4).

Regras do plano:
  - o Stockfish NUNCA escolhe jogadas dos agentes; so avalia depois;
  - configuracao fixa: 1 thread, hash 32 MB, profundidade 14, sem livro;
  - melhor jogada = analise de todos os lances legais; jogada do agente =
    analise restrita a esse lance (root_moves);
  - regret = max(0, melhor - agente), em centipeoes, do ponto de vista de
    quem joga; resultados de mate sao tratados a parte.

IMPORTANTE: rode a avaliacao DEPOIS dos experimentos (nunca junto), senao o
Stockfish disputa a CPU e atrapalha as medidas de tempo.

Uso:
  python -m experiments.reference evaluate --stockfish C:/caminho/stockfish.exe \
      --positions experiments/data/positions.csv \
      --decisions experiments/results/decisions.csv \
      --quality experiments/results/quality.csv
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from experiments.results_storage import (QualityRecord, load_rows, quality_table)

DEPTH = 14
THREADS = 1
HASH_MB = 32


@dataclass
class Score:
    cp: int | None       # centipeoes (None se for mate)
    mate: int | None     # lances ate o mate: >0 quem joga da mate, <0 quem joga leva mate


def compute_regret(best: Score, agent: Score) -> tuple[float | None, str]:
    """Regret em centipeoes e a classe do resultado. Funcao pura (testavel)."""
    if best.mate is not None:
        if best.mate > 0:
            if agent.mate is not None and agent.mate > 0:
                return 0.0, "mate_kept"          # manteve o mate forcado
            return None, "mate_missed"           # perdeu um mate forcado
        # melhor jogada ja perde por mate
        if agent.mate is not None and agent.mate < 0:
            return 0.0, "lost_both"
        return None, "inconsistent"
    if agent.mate is not None:
        if agent.mate < 0:
            return None, "mate_blunder"          # saiu de posicao normal para levar mate
        return None, "inconsistent"              # agente "melhor" que a referencia
    assert best.cp is not None and agent.cp is not None
    return float(max(0, best.cp - agent.cp)), "cp"


def _score_of(info, board) -> Score:
    pov = info["score"].pov(board.turn)
    if pov.is_mate():
        return Score(None, pov.mate())
    return Score(pov.score(), None)


class StockfishReference:
    def __init__(self, path: str, depth: int = DEPTH):
        import chess.engine
        self.path = path
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": THREADS, "Hash": HASH_MB})
        self.name = self.engine.id.get("name", "stockfish")
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self._best_cache: dict[str, tuple[str, Score]] = {}
        self._move_cache: dict[tuple[str, str], Score] = {}

    def _limit(self):
        import chess.engine
        return chess.engine.Limit(depth=self.depth)

    def best(self, position_id: str, board) -> tuple[str, Score]:
        if position_id not in self._best_cache:
            # game=<id unico> -> o motor recomeca "limpo" a cada analise (ordem nao importa)
            info = self.engine.analyse(board, self._limit(), game=f"{position_id}:best")
            self._best_cache[position_id] = (info["pv"][0].uci(), _score_of(info, board))
        return self._best_cache[position_id]

    def move_score(self, position_id: str, board, move) -> Score:
        key = (position_id, move.uci())
        if key not in self._move_cache:
            info = self.engine.analyse(board, self._limit(), root_moves=[move],
                                       game=f"{position_id}:{move.uci()}")
            self._move_cache[key] = _score_of(info, board)
        return self._move_cache[key]

    def close(self) -> None:
        self.engine.quit()


def evaluate_decisions(ref: StockfishReference, positions_csv: str, decisions_csv: str,
                       quality_csv: str) -> int:
    """Avalia todas as decisoes de posicao (run_kind == 'time') que ainda nao tem qualidade."""
    import chess
    from experiments.dataset import load_positions

    positions = {p.position_id: p for p in load_positions(positions_csv)}
    done = {(r["experiment_id"], r["run_id"]) for r in load_rows(quality_csv)}
    n = 0
    with quality_table(quality_csv) as table:
        for row in load_rows(decisions_csv):
            if row["run_kind"] != "time" or row["status"] != "ok":
                continue
            if (row["experiment_id"], row["run_id"]) in done:
                continue
            pos = positions[row["position_id"]]
            board = pos.to_board()
            best_uci, best_score = ref.best(pos.position_id, board)
            move = chess.Move.from_uci(row["action_uci"])
            agent_score = ref.move_score(pos.position_id, board, move)
            regret, klass = compute_regret(best_score, agent_score)
            table.write(QualityRecord(
                experiment_id=row["experiment_id"], run_id=int(row["run_id"]),
                position_id=pos.position_id, agent=row["agent"], action_uci=row["action_uci"],
                ref_best_uci=best_uci, top1=(row["action_uci"] == best_uci),
                ref_best_cp=best_score.cp, ref_best_mate=best_score.mate,
                agent_cp=agent_score.cp, agent_mate=agent_score.mate,
                regret_cp=regret, decisive_class=klass,
                ref_engine=f"{ref.name} sha256={ref.sha256[:12]}", ref_depth=ref.depth))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="avalia as decisoes ja gravadas")
    e.add_argument("--stockfish", required=True)
    e.add_argument("--positions", default="experiments/data/positions.csv")
    e.add_argument("--decisions", default="experiments/results/decisions.csv")
    e.add_argument("--quality", default="experiments/results/quality.csv")
    a = ap.parse_args()
    ref = StockfishReference(a.stockfish)
    try:
        print(f"Motor: {ref.name} (sha256 {ref.sha256})")
        n = evaluate_decisions(ref, a.positions, a.decisions, a.quality)
        print(f"{n} decisoes avaliadas")
    finally:
        ref.close()


if __name__ == "__main__":
    main()
