"""
Testes da infraestrutura de experimentos (nao dependem de python-chess nem do Stockfish).

Rodar na raiz do projeto:   python -m pytest tests/test_experiments.py
Sem pytest:                 python tests/test_experiments.py
"""
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analysis
from experiments.dataset import NotEnoughPositions, pool_of, select_positions
from experiments.metrics import budget_overrun, budget_parts, is_timeout, measure
from experiments.reference import Score, compute_regret
from experiments.results_storage import (DecisionRecord, GameRecord, QualityRecord, decision_table,
                                         game_table, load_rows, next_run_id, quality_table)
from experiments.runner import Budget, run_decision, run_positions


# ----------------------------------------------------------------------------
# Falsos para simular o motor e os agentes
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class FakeMove:
    u: str

    def uci(self):
        return self.u


class FakeBoard:
    def __init__(self):
        self.move_stack = []
        self.legal_moves = [FakeMove("e2e4"), FakeMove("d2d4")]

    def fen(self):
        return "fen-" + str(len(self.move_stack))

    def can_claim_draw(self):
        return False


@dataclass
class FakeDecision:
    action: object
    nodes_visited: int = 10
    nodes_expanded: int = 4
    transitions: int = 20
    max_depth_reached: int = 3
    extra: dict = field(default_factory=dict)


class FakeAgent:
    def __init__(self, mode="ok", seed=None, name="minimax", sleep=0.0):
        self.mode, self.seed, self.name, self.sleep = mode, seed, name, sleep

    def choose_action(self, board, budget):
        time.sleep(self.sleep)
        if self.mode == "error":
            raise RuntimeError("quebrou")
        if self.mode == "mutate":
            board.move_stack.append("x")
        if self.mode == "illegal":
            return FakeDecision(FakeMove("a1a8"))
        extra = {"alpha_beta_prunes": 5, "complete_depth": 3, "context": "C1",
                 "chosen_algorithm": "minimax", "chave_nova": 1}
        if self.name == "gpn_mcts":
            extra = {"iterations": 50, "playouts": 50, "proof_updates": 7, "pnrank_sorts": 7}
        # mais iteracoes/nos em orcamentos maiores nao importam aqui
        move = FakeMove("e2e4" if (self.seed or 0) % 2 == 0 else "d2d4")
        return FakeDecision(move, extra=extra)


@dataclass
class FakePosition:
    position_id: str
    set: str
    context: str

    def to_board(self):
        return FakeBoard()


def factories():
    return {
        "minimax": lambda seed: FakeAgent(name="minimax"),
        "gpn_mcts": lambda seed: FakeAgent(seed=seed, name="gpn_mcts"),
        "hybrid": lambda seed: FakeAgent(seed=seed, name="minimax"),
    }


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def test_budget_helpers():
    t, c = Budget(max_time_ms=100), Budget(max_transitions=500)
    assert budget_parts(t) == ("time", 100.0)
    assert budget_parts(c) == ("transitions", 500.0)
    assert budget_overrun(t, 120, 10) and not budget_overrun(t, 90, 10)
    assert budget_overrun(c, 1, 501) and not budget_overrun(c, 1, 500)
    assert is_timeout(t, 201) and not is_timeout(t, 150) and not is_timeout(c, 10 ** 6)
    try:
        budget_parts(Budget(max_time_ms=1, max_transitions=1))
        raise AssertionError("deveria exigir UM limite")
    except ValueError:
        pass


def test_measure_time_and_memory():
    with measure(track_memory=True) as m:
        _ = [0] * 100000
    assert m.elapsed_ms > 0 and m.peak_memory_kb and m.peak_memory_kb > 0
    with measure() as m2:
        pass
    assert m2.peak_memory_kb is None


# ----------------------------------------------------------------------------
# run_decision: o contrato e conferido de fora
# ----------------------------------------------------------------------------
def test_run_decision_statuses():
    b = Budget(max_time_ms=10)
    assert run_decision(FakeAgent(), FakeBoard(), b).status == "ok"
    assert run_decision(FakeAgent("illegal"), FakeBoard(), b).status == "illegal_action"
    o = run_decision(FakeAgent("error"), FakeBoard(), b)
    assert o.status == "error" and "quebrou" in o.error
    o = run_decision(FakeAgent("mutate"), FakeBoard(), b)
    assert o.status == "error" and "modificado" in o.error
    o = run_decision(FakeAgent(sleep=0.05), FakeBoard(), b)
    assert o.status == "timeout" and o.overrun
    o = run_decision(FakeAgent(sleep=0.015), FakeBoard(), b)
    assert o.status == "ok" and o.overrun          # estourou, mas nao passou do dobro


# ----------------------------------------------------------------------------
# results_storage
# ----------------------------------------------------------------------------
def _decision(**over):
    base = dict(
        experiment_id="e", run_id=1, run_kind="time", position_id="cal_C1_01",
        position_set="calibration", context="C1", game_id="", ply=None, agent="minimax", seed=None,
        repetition=0, budget_type="time", budget_value=100.0, git_commit="abc", action_uci="e2e4",
        action_changed=None, nodes_visited=10, nodes_expanded=4, tree_nodes_created=None,
        transitions=20, max_depth_reached=3, elapsed_ms=1.0, cpu_ms=1.0, peak_memory_kb=None,
        budget_overrun=False, status="ok")
    base.update(over)
    return DecisionRecord(**base)


def test_storage_validation_and_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "decisions.csv"
        with decision_table(p) as t:
            t.write(_decision())
            for bad in (dict(status="empate"), dict(nodes_expanded=99), dict(agent="stockfish"),
                        dict(peak_memory_kb=5.0), dict(budget_type="nodes")):
                try:
                    t.write(_decision(**bad))
                    raise AssertionError(f"deveria recusar {bad}")
                except ValueError:
                    pass
        rows = load_rows(p)
        assert len(rows) == 1 and rows[0]["action_uci"] == "e2e4"
        assert next_run_id(p) == 2
        with decision_table(p) as t:                       # reabrir nao duplica o cabecalho
            t.write(_decision(run_id=2))
        assert len(load_rows(p)) == 2


def test_failed_game_is_never_a_draw():
    with tempfile.TemporaryDirectory() as d:
        with game_table(Path(d) / "g.csv") as t:
            base = dict(experiment_id="e", game_id="g1", matchup="minimax_vs_gpn_mcts",
                        white="minimax", black="gpn_mcts", seed_white=None, seed_black=2026,
                        budget_type="time", budget_value=300.0, plies=10, failures=1,
                        points_white=None, points_black=None, pgn_path="x", git_commit="abc")
            t.write(GameRecord(result="*", status="failure", termination="error:gpn_mcts", **base))
            try:
                t.write(GameRecord(result="1/2-1/2", status="failure", termination="x", **base))
                raise AssertionError("falha nao pode virar empate")
            except ValueError:
                pass


# ----------------------------------------------------------------------------
# reference: perda em centipeoes
# ----------------------------------------------------------------------------
def test_regret():
    assert compute_regret(Score(50, None), Score(20, None)) == (30.0, "cp")
    assert compute_regret(Score(50, None), Score(80, None)) == (0.0, "cp")      # nunca negativa
    assert compute_regret(Score(None, 3), Score(None, 5)) == (0.0, "mate_kept")
    assert compute_regret(Score(None, 3), Score(120, None)) == (None, "mate_missed")
    assert compute_regret(Score(30, None), Score(None, -2)) == (None, "mate_blunder")
    assert compute_regret(Score(None, -4), Score(None, -2)) == (0.0, "lost_both")


# ----------------------------------------------------------------------------
# dataset: selecao e divisao
# ----------------------------------------------------------------------------
def _candidates(n_games=200):
    out = []
    for i in range(n_games):
        for c in ("C1", "C2", "C3", "C4", "C5", "C6"):
            out.append(dict(game_id=f"g{i}", context=c, key=f"{c}-{i}", ply=i))
    return out


def test_selection_and_split():
    sel = select_positions(_candidates())
    assert len(sel) == 120
    for c in ("C1", "C2", "C3", "C4", "C5", "C6"):
        for s in ("calibration", "test"):
            assert sum(1 for x in sel if x["context"] == c and x["set"] == s) == 10
    cal = {x["game_id"] for x in sel if x["set"] == "calibration"}
    tst = {x["game_id"] for x in sel if x["set"] == "test"}
    assert not (cal & tst), "partida nas duas bases"
    assert len({x["position_id"] for x in sel}) == 120
    assert select_positions(_candidates()) == sel                          # deterministico
    assert pool_of("g1") == pool_of("g1")


def test_selection_skips_duplicates_and_reports_shortage():
    dup = [dict(game_id=f"g{i}", context="C1", key="mesma-posicao", ply=1) for i in range(500)]
    try:
        select_positions(dup)
        raise AssertionError("deveria faltar posicoes")
    except NotEnoughPositions as e:
        assert "C2" in str(e)


# ----------------------------------------------------------------------------
# runner + analysis de ponta a ponta (com agentes falsos)
# ----------------------------------------------------------------------------
def _positions():
    ctxs = ["C1", "C2", "C3", "C4", "C5", "C6"]
    return ([FakePosition(f"cal_{c}_{i}", "calibration", c) for c in ctxs for i in range(2)] +
            [FakePosition(f"tes_{c}_{i}", "test", c) for c in ctxs for i in range(2)])


def test_runner_end_to_end_and_resume():
    pos = _positions()[:2]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "decisions.csv"
        with decision_table(p) as t:
            n = run_positions(pos, ["minimax", "gpn_mcts"], "time", (100, 300), table=t,
                              experiment_id="x", reps=2, factories=factories(), commit="abc",
                              verbose=False)
        assert n == 2 * 2 * (2 + 3 * 2)                                     # pos x orc x (minimax + 3 seeds)
        rows = load_rows(p)
        assert all(r["status"] == "ok" for r in rows)
        mm = [r for r in rows if r["agent"] == "minimax" and r["repetition"] == "1"]
        assert mm and all(r["action_changed"] == "False" for r in mm)       # Minimax estavel
        assert [r for r in rows if r["agent"] == "gpn_mcts"][0]["iterations"] == "50"
        assert "chave_nova" in [r for r in rows if r["agent"] == "minimax"][0]["extra_json"]
        with decision_table(p) as t:                                        # retomar nao repete
            assert run_positions(pos, ["minimax", "gpn_mcts"], "time", (100, 300), table=t,
                                 experiment_id="x", reps=2, factories=factories(),
                                 commit="abc", verbose=False) == 0
        with decision_table(p) as t:                                        # rodada de memoria
            run_positions(pos, ["minimax"], "time", (100,), table=t, experiment_id="m",
                          run_kind="memory", reps=5, factories=factories(), commit="abc",
                          verbose=False)
        mem = [r for r in load_rows(p) if r["run_kind"] == "memory"]
        assert len(mem) == 2 and all(float(r["peak_memory_kb"]) > 0 for r in mem)


def test_analysis_pipeline_smoke():
    pos = _positions()
    with tempfile.TemporaryDirectory() as d:
        res, plots = Path(d) / "results", Path(d) / "plots"
        with decision_table(res / "decisions.csv") as t:
            for btype, vals in (("time", (100, 300, 1000)), ("transitions", (500, 2000))):
                run_positions(pos, ["minimax", "gpn_mcts", "hybrid"], btype, vals, table=t,
                              experiment_id=f"e_{btype}", reps=1, factories=factories(),
                              commit="abc", verbose=False)
            run_positions(pos[:4], ["minimax", "gpn_mcts"], "time", (300,), table=t,
                          experiment_id="mem", run_kind="memory", factories=factories(),
                          commit="abc", verbose=False)
        with quality_table(res / "quality.csv") as q:
            for r in load_rows(res / "decisions.csv"):
                if r["run_kind"] != "time":
                    continue
                q.write(QualityRecord(
                    experiment_id=r["experiment_id"], run_id=int(r["run_id"]),
                    position_id=r["position_id"], agent=r["agent"], action_uci=r["action_uci"],
                    ref_best_uci="e2e4", top1=(r["action_uci"] == "e2e4"), ref_best_cp=30,
                    ref_best_mate=None, agent_cp=10, agent_mate=None,
                    regret_cp=0.0 if r["action_uci"] == "e2e4" else 20.0, decisive_class="cp",
                    ref_engine="fake", ref_depth=14))
        with game_table(res / "games.csv") as g:
            for i, (res_, w, b) in enumerate([("1-0", "minimax", "gpn_mcts"),
                                              ("1/2-1/2", "gpn_mcts", "minimax"),
                                              ("0-1", "minimax", "gpn_mcts")]):
                g.write(GameRecord(
                    experiment_id="g", game_id=f"g{i}", matchup="minimax_vs_gpn_mcts", white=w,
                    black=b, seed_white=None, seed_black=2026, budget_type="time",
                    budget_value=300.0, result=res_, status="completed", termination="checkmate",
                    plies=40, failures=0, points_white=1.0, points_black=0.0, pgn_path="x",
                    git_commit="abc"))
        analysis.run_all(res, plots)
        names = {p.name for p in plots.iterdir()}
        for prefix in ("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_"):
            assert any(n.startswith(prefix) for n in names), f"faltou o grafico {prefix}"
        cal = load_rows(res / "calibration_table.csv")
        assert cal and {"agent", "context", "tentativas", "acertos"} <= set(cal[0])
        assert {r["agent"] for r in cal} == {"minimax", "gpn_mcts"}           # hibrida nao calibra
        assert (res / "empirical_summary.csv").exists()
        fixed = [r for r in load_rows(res / "quality_per_position.csv") if r["agent"] == "fixed_policy"]
        assert fixed, "politica fixa nao foi derivada"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\n{len(tests)} testes passaram")
