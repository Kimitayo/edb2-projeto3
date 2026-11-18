
import pytest
import chess

import algorithms.hybrid as hybrid_module

from algorithms.base import Budget, Decision
from algorithms.hybrid import (
    CONTEXTS,
    normalize_algorithm,
    normalize_budget_key,
    budget_key_from_budget,
    laplace_probability,
    expected_utility,
    parse_correct,
    build_probability_table,
    load_calibration_csv,
    select_algorithm,
    select_fixed_algorithm,
    HybridAgent,
)


# =========================================================
# 1. NORMALIZAÇÃO DOS NOMES DOS ALGORITMOS
# =========================================================

@pytest.mark.parametrize(
    "name, expected",
    [
        ("minimax", "minimax"),
        ("MINIMAX", "minimax"),
        ("alphabeta", "minimax"),
        ("alpha-beta", "minimax"),
        ("alpha beta", "minimax"),
        ("gpn_mcts", "gpn_mcts"),
        ("GPN-MCTS", "gpn_mcts"),
        ("gpnmcts", "gpn_mcts"),
    ],
)
def test_normalize_algorithm(name, expected):
    assert normalize_algorithm(name) == expected


def test_normalize_algorithm_invalid():
    with pytest.raises(ValueError):
        normalize_algorithm("stockfish")


# =========================================================
# 2. NORMALIZAÇÃO DOS ORÇAMENTOS
# =========================================================

@pytest.mark.parametrize(
    "value, expected",
    [
        ("100ms", "100ms"),
        (" 100 MS ", "100ms"),
        ("300ms", "300ms"),
        ("1000ms", "1000ms"),
        ("500_transitions", "500_transitions"),
        ("500-transitions", "500_transitions"),
        ("2000_transitions", "2000_transitions"),
        ("5000_transitions", "5000_transitions"),
    ],
)
def test_normalize_budget_key(value, expected):
    assert normalize_budget_key(value) == expected


def test_normalize_budget_key_invalid():
    with pytest.raises(ValueError):
        normalize_budget_key("250ms")


# =========================================================
# 3. IDENTIFICAÇÃO DO ORÇAMENTO RECEBIDO
# =========================================================

@pytest.mark.parametrize(
    "budget, expected",
    [
        (Budget(max_time_s=0.1), "100ms"),
        (Budget(max_time_s=0.3), "300ms"),
        (Budget(max_time_s=1.0), "1000ms"),
        (Budget(max_transitions=500), "500_transitions"),
        (Budget(max_transitions=2000), "2000_transitions"),
        (Budget(max_transitions=5000), "5000_transitions"),
    ],
)
def test_budget_key_from_budget(budget, expected):
    assert budget_key_from_budget(budget) == expected


@pytest.mark.parametrize(
    "budget",
    [
        Budget(),
        Budget(max_time_s=0.0),
        Budget(max_time_s=-0.1),
        Budget(max_transitions=0),
        Budget(max_transitions=-1),
        Budget(max_time_s=0.1, max_transitions=500),
    ],
)
def test_budget_key_from_budget_invalid(budget):
    with pytest.raises(ValueError):
        budget_key_from_budget(budget)


# =========================================================
# 4. SUAVIZAÇÃO DE LAPLACE
# =========================================================

def test_laplace_probability():
    assert laplace_probability(8, 10) == 0.75


@pytest.mark.parametrize(
    "correct, attempts, expected",
    [
        (0, 10, 1 / 12),
        (10, 10, 11 / 12),
        (2.5, 3, 0.7),
        (0, 0, 0.5),
    ],
)
def test_laplace_probability_edge_cases(
    correct,
    attempts,
    expected,
):
    result = laplace_probability(correct, attempts)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "correct, attempts",
    [
        (-1, 10),
        (11, 10),
        (0, -1),
        (1.5, 1),
    ],
)
def test_laplace_probability_invalid(correct, attempts):
    with pytest.raises(ValueError):
        laplace_probability(correct, attempts)


# =========================================================
# 5. UTILIDADE ESPERADA
# =========================================================

def test_expected_utility():
    assert expected_utility(0.75) == 0.75


@pytest.mark.parametrize(
    "probability",
    [0.0, 0.25, 0.5, 1.0],
)
def test_expected_utility_valid(probability):
    assert expected_utility(probability) == probability


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1],
)
def test_expected_utility_invalid(probability):
    with pytest.raises(ValueError):
        expected_utility(probability)


# =========================================================
# 6. CONVERSÃO DOS RESULTADOS DE CALIBRAÇÃO
# =========================================================

@pytest.mark.parametrize(
    "value, expected",
    [
        (True, 1.0),
        (False, 0.0),
        ("true", 1.0),
        ("TRUE", 1.0),
        ("yes", 1.0),
        ("false", 0.0),
        ("no", 0.0),
        ("0", 0.0),
        ("1", 1.0),
        ("0.5", 0.5),
    ],
)
def test_parse_correct(value, expected):
    assert parse_correct(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "2",
        "-0.1",
        "nan",
        "invalid",
    ],
)
def test_parse_correct_invalid(value):
    with pytest.raises(ValueError):
        parse_correct(value)


# =========================================================
# 7. CONSTRUÇÃO DA TABELA DE PROBABILIDADES
# =========================================================

def test_build_probability_table_aggregates_results_by_position():

    records = [
        # Minimax: posição P1
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "minimax",
            "correct": 1,
        },
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "minimax",
            "correct": 0,
        },

        # Minimax: posição P2
        {
            "scenario_id": "P2",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "minimax",
            "correct": 0,
        },

        # GPN-MCTS: posição P1, três seeds
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 1,
        },
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 0,
        },
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 1,
        },

        # GPN-MCTS: posição P2, três seeds
        {
            "scenario_id": "P2",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 0,
        },
        {
            "scenario_id": "P2",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 0,
        },
        {
            "scenario_id": "P2",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 0,
        },
    ]

    table = build_probability_table(records)

    # Minimax:
    # P1 = 0.5
    # P2 = 0.0
    # Laplace = (0.5 + 1) / (2 + 2) = 0.375

    assert table["100ms"]["C1"]["minimax"] == pytest.approx(0.375)

    # GPN-MCTS:
    # P1 = 2/3
    # P2 = 0
    # Laplace = (2/3 + 1) / (2 + 2) = 5/12

    assert table["100ms"]["C1"]["gpn_mcts"] == pytest.approx(5 / 12)


def test_build_probability_table_separates_contexts_budgets_and_algorithms():

    records = [
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "minimax",
            "correct": 1,
        },
        {
            "scenario_id": "P1",
            "context": "C1",
            "budget": "100ms",
            "algorithm": "gpn_mcts",
            "correct": 0,
        },
        {
            "scenario_id": "P2",
            "context": "C2",
            "budget": "100ms",
            "algorithm": "minimax",
            "correct": 0,
        },
        {
            "scenario_id": "P3",
            "context": "C1",
            "budget": "300ms",
            "algorithm": "minimax",
            "correct": 0,
        },
    ]

    table = build_probability_table(records)

    assert table["100ms"]["C1"]["minimax"] == pytest.approx(2 / 3)
    assert table["100ms"]["C1"]["gpn_mcts"] == pytest.approx(1 / 3)
    assert table["100ms"]["C2"]["minimax"] == pytest.approx(1 / 3)
    assert table["300ms"]["C1"]["minimax"] == pytest.approx(1 / 3)


def test_build_probability_table_empty_records():
    assert build_probability_table([]) == {}


@pytest.mark.parametrize(
    "record, exception",
    [
        (
            {
                "scenario_id": "P1",
                "context": "C7",
                "budget": "100ms",
                "algorithm": "minimax",
                "correct": 1,
            },
            ValueError,
        ),
        (
            {
                "scenario_id": "P1",
                "context": "C1",
                "budget": "250ms",
                "algorithm": "minimax",
                "correct": 1,
            },
            ValueError,
        ),
        (
            {
                "scenario_id": "P1",
                "context": "C1",
                "budget": "100ms",
                "algorithm": "stockfish",
                "correct": 1,
            },
            ValueError,
        ),
        (
            {
                "scenario_id": "P1",
                "context": "C1",
                "budget": "100ms",
                "algorithm": "minimax",
                "correct": 2,
            },
            ValueError,
        ),
        (
            {
                "context": "C1",
                "budget": "100ms",
                "algorithm": "minimax",
                "correct": 1,
            },
            KeyError,
        ),
    ],
)
def test_build_probability_table_rejects_invalid_records(
    record,
    exception,
):
    with pytest.raises(exception):
        build_probability_table([record])


# =========================================================
# 8. LEITURA DO CSV DE CALIBRAÇÃO
# =========================================================

def test_load_calibration_csv(tmp_path):

    csv_content = (
        "scenario_id,context,budget,algorithm,correct\n"
        "P1,C1,100ms,minimax,1\n"
        "P2,C1,100ms,minimax,0\n"
        "P1,C1,100ms,gpn_mcts,0\n"
    )

    csv_path = tmp_path / "calibration.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    table = load_calibration_csv(csv_path)

    assert table["100ms"]["C1"]["minimax"] == pytest.approx(0.5)
    assert table["100ms"]["C1"]["gpn_mcts"] == pytest.approx(1 / 3)


def test_load_calibration_csv_file_not_found(tmp_path):

    csv_path = tmp_path / "inexistent.csv"

    with pytest.raises(FileNotFoundError):
        load_calibration_csv(csv_path)


# =========================================================
# 9. SELEÇÃO DO ALGORITMO PELA MEU
# =========================================================

def test_select_minimax_when_it_has_higher_utility():

    probabilities = {
        "minimax": 0.85,
        "gpn_mcts": 0.70,
    }

    assert select_algorithm(probabilities) == "minimax"


def test_select_gpn_mcts_when_it_has_higher_utility():

    probabilities = {
        "minimax": 0.65,
        "gpn_mcts": 0.90,
    }

    assert select_algorithm(probabilities) == "gpn_mcts"


def test_select_minimax_when_utilities_are_equal():

    probabilities = {
        "minimax": 0.80,
        "gpn_mcts": 0.80,
    }

    assert select_algorithm(probabilities) == "minimax"


@pytest.mark.parametrize(
    "probabilities",
    [
        {"minimax": 0.80},
        {"gpn_mcts": 0.80},
    ],
)
def test_select_algorithm_missing_probability(probabilities):

    with pytest.raises(KeyError):
        select_algorithm(probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"minimax": -0.1, "gpn_mcts": 0.8},
        {"minimax": 0.8, "gpn_mcts": 1.5},
    ],
)
def test_select_algorithm_invalid_probability(probabilities):

    with pytest.raises(ValueError):
        select_algorithm(probabilities)


# =========================================================
# 10. POLÍTICA FIXA DE REFERÊNCIA
# =========================================================

def make_fixed_probability_table(minimax, gpn_mcts):

    return {
        "100ms": {
            context: {
                "minimax": minimax,
                "gpn_mcts": gpn_mcts,
            }
            for context in CONTEXTS
        }
    }


def test_select_fixed_algorithm_prefers_gpn_mcts():

    table = make_fixed_probability_table(
        minimax=0.65,
        gpn_mcts=0.85,
    )

    assert select_fixed_algorithm(table, "100ms") == "gpn_mcts"


def test_select_fixed_algorithm_prefers_minimax():

    table = make_fixed_probability_table(
        minimax=0.85,
        gpn_mcts=0.65,
    )

    assert select_fixed_algorithm(table, "100ms") == "minimax"


def test_select_fixed_algorithm_tie_prefers_minimax():

    table = make_fixed_probability_table(
        minimax=0.80,
        gpn_mcts=0.80,
    )

    assert select_fixed_algorithm(table, "100ms") == "minimax"


def test_select_fixed_algorithm_missing_context():

    table = make_fixed_probability_table(0.8, 0.7)

    del table["100ms"]["C3"]

    with pytest.raises(KeyError):
        select_fixed_algorithm(table, "100ms")


def test_select_fixed_algorithm_invalid_budget():

    table = make_fixed_probability_table(0.8, 0.7)

    with pytest.raises(ValueError):
        select_fixed_algorithm(table, "250ms")


# =========================================================
# 11. AGENTES SIMULADOS PARA TESTAR HYBRIDAGENT
# =========================================================

class StubAgent:

    def __init__(self, action):
        self.action = action
        self.calls = []

    def choose_action(self, state, budget):

        self.calls.append({
            "state": state,
            "budget": budget,
        })

        return Decision(
            action=self.action,
            nodes_expanded=10,
            nodes_visited=20,
            max_depth_reached=4,
            extra={"stub": True},
        )



def make_hybrid_probability_table(
    minimax=0.8,
    gpn_mcts=0.6,
):
    budgets = (
        "100ms",
        "300ms",
        "1000ms",
        "500_transitions",
        "2000_transitions",
        "5000_transitions",
    )

    return {
        budget: {
            context: {
                "minimax": minimax,
                "gpn_mcts": gpn_mcts,
            }
            for context in CONTEXTS
        }
        for budget in budgets
    }



def prepare_hybrid_mocks(
    monkeypatch,
    actions,
    clock_values,
    transition_values,
):
    clock = iter(clock_values)
    transitions = iter(transition_values)

    monkeypatch.setattr(
        hybrid_module,
        "perf_counter",
        lambda: next(clock),
    )

    monkeypatch.setattr(
        hybrid_module,
        "transition_count",
        lambda: next(transitions),
    )

    monkeypatch.setattr(
        hybrid_module,
        "legal_actions",
        lambda state: actions,
    )

    monkeypatch.setattr(
        hybrid_module,
        "state_features",
        lambda state, received_actions: {
            "context": "C1",
            "phase": "opening",
            "tactical": False,
        },
    )


# =========================================================
# 12. EXECUÇÃO DO HYBRIDAGENT
# =========================================================

def test_hybrid_agent_executes_only_minimax(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(0.8, 0.6),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0, 1.01, 1.02],
        transition_values=[0, 0, 0],
    )

    decision = agent.choose_action(
        board,
        Budget(max_time_s=0.1, max_depth=5),
    )

    assert decision.action == action

    assert len(minimax.calls) == 1
    assert len(gpn_mcts.calls) == 0

    assert decision.extra["hybrid"]["selected_algorithm"] == "minimax"
    assert decision.extra["hybrid"]["context"] == "C1"

    assert decision.extra["agent_extra"] == {"stub": True}


def test_hybrid_agent_executes_only_gpn_mcts(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(0.6, 0.9),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0, 1.01, 1.02],
        transition_values=[0, 0, 0],
    )

    decision = agent.choose_action(
        board,
        Budget(max_time_s=0.1),
    )

    assert decision.action == action

    assert len(minimax.calls) == 0
    assert len(gpn_mcts.calls) == 1

    assert decision.extra["hybrid"]["selected_algorithm"] == "gpn_mcts"


# =========================================================
# 13. DESCONTO DO ORÇAMENTO COMPUTACIONAL
# =========================================================

def test_hybrid_agent_deducts_selection_time(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0, 1.01, 1.02],
        transition_values=[0, 0, 0],
    )

    agent.choose_action(
        board,
        Budget(max_time_s=0.1, max_depth=5),
    )

    remaining_budget = minimax.calls[0]["budget"]

    assert remaining_budget.max_time_s == pytest.approx(0.09)
    assert remaining_budget.max_transitions is None
    assert remaining_budget.max_depth == 5


def test_hybrid_agent_deducts_selection_transitions(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0, 1.01, 1.02],
        transition_values=[10, 12, 15],
    )

    decision = agent.choose_action(
        board,
        Budget(max_transitions=500, max_depth=4),
    )

    remaining_budget = minimax.calls[0]["budget"]

    assert remaining_budget.max_transitions == 498
    assert remaining_budget.max_time_s is None
    assert remaining_budget.max_depth == 4

    assert decision.extra["hybrid"]["selection_transitions"] == 2
    assert decision.extra["hybrid"]["total_transitions"] == 5
    assert decision.transitions == 5


# =========================================================
# 14. FALLBACK QUANDO O ORÇAMENTO É ESGOTADO
# =========================================================

def test_hybrid_agent_fallback_when_time_is_exhausted(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0, 1.11],
        transition_values=[0, 0],
    )

    decision = agent.choose_action(
        board,
        Budget(max_time_s=0.1),
    )

    assert decision.action == action

    assert len(minimax.calls) == 0
    assert len(gpn_mcts.calls) == 0

    assert decision.extra["hybrid"]["fallback"] == "budget_exhausted"


def test_hybrid_agent_fallback_when_transitions_are_exhausted(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0, 1.01],
        transition_values=[0, 500],
    )

    decision = agent.choose_action(
        board,
        Budget(max_transitions=500),
    )

    assert decision.action == action

    assert len(minimax.calls) == 0
    assert len(gpn_mcts.calls) == 0

    assert decision.extra["hybrid"]["fallback"] == "budget_exhausted"
    assert decision.transitions == 500


# =========================================================
# 15. TRATAMENTO DE POSIÇÕES E CALIBRAÇÃO INVÁLIDAS
# =========================================================

def test_hybrid_agent_rejects_terminal_position(monkeypatch):

    minimax = StubAgent(None)
    gpn_mcts = StubAgent(None)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[],
        clock_values=[1.0],
        transition_values=[0],
    )

    with pytest.raises(ValueError):
        agent.choose_action(
            chess.Board(),
            Budget(max_time_s=0.1),
        )


def test_hybrid_agent_rejects_missing_calibration(monkeypatch):

    board = chess.Board()
    action = chess.Move.from_uci("e2e4")

    minimax = StubAgent(action)
    gpn_mcts = StubAgent(action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        {},
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[action],
        clock_values=[1.0],
        transition_values=[0],
    )

    with pytest.raises(KeyError):
        agent.choose_action(
            board,
            Budget(max_time_s=0.1),
        )


def test_hybrid_agent_rejects_illegal_action(monkeypatch):

    board = chess.Board()

    legal_action = chess.Move.from_uci("e2e4")
    illegal_action = chess.Move.from_uci("a1a8")

    minimax = StubAgent(illegal_action)
    gpn_mcts = StubAgent(legal_action)

    agent = HybridAgent(
        minimax,
        gpn_mcts,
        make_hybrid_probability_table(),
    )

    prepare_hybrid_mocks(
        monkeypatch,
        actions=[legal_action],
        clock_values=[1.0, 1.01],
        transition_values=[0, 0],
    )

    with pytest.raises(ValueError):
        agent.choose_action(
            board,
            Budget(max_time_s=0.1),
        )
