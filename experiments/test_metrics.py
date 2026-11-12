"""Testes rapidos: rode com  python -m pytest experiments  (ou python experiments/test_metrics.py)"""
import tempfile, os
from experiments.metrics import Budget, SearchCounter, measure, BattleAggregator
from experiments.results_storage import RunRecord, ResultsWriter, load_results, COLUMNS


def fake_search(counter):
    """Algoritmo de mentira: visita nos ate o orcamento acabar."""
    while not counter.exhausted():
        counter.visit(depth=3)
        if counter.visited % 2 == 0:
            counter.expand()


def test_node_budget_is_respected():
    c = SearchCounter(Budget(max_nodes=100))
    fake_search(c)
    assert c.visited == 100
    assert c.expanded <= c.visited


def test_time_and_memory_measurement():
    c = SearchCounter(Budget(max_nodes=1000))
    with measure(track_memory=True) as m:
        fake_search(c)
    assert m.wall_time_ms > 0
    assert m.peak_memory_kb is not None


def test_csv_roundtrip():
    agg = BattleAggregator()
    c = SearchCounter(Budget(max_nodes=50))
    with measure() as m:
        fake_search(c)
    agg.add(c, m)
    rec = RunRecord(
        experiment_id="teste", run_id=1, scenario_id="sc_0001", scenario_seed=1,
        run_seed=1, repetition=0, git_commit="abc123", algorithm="alphabeta",
        opponent="mcts", budget_type="nodes", budget_value=50,
        scenario_difficulty="small", winner="A", turns=10,
        hp_remaining_a=30, hp_remaining_b=0, decisions=agg.decisions,
        nodes_visited=agg.nodes_visited, nodes_expanded=agg.nodes_expanded,
        max_depth=agg.max_depth, total_time_ms=agg.total_time_ms,
        mean_decision_time_ms=agg.mean_decision_time_ms,
        peak_memory_kb=None, budget_exhausted_count=agg.budget_exhausted_count,
    )
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.csv")
        with ResultsWriter(p) as w:
            w.write(rec)
        rows = load_results(p)
    assert len(rows) == 1 and list(rows[0].keys()) == COLUMNS


if __name__ == "__main__":
    test_node_budget_is_respected(); test_time_and_memory_measurement(); test_csv_roundtrip()
    print("todos os testes passaram")
