
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from algorithms.base import Agent, Budget, Decision
from game import legal_actions, transition_count
from game.features import state_features


# Contextos definidos no projeto.
CONTEXTS = ("C1", "C2", "C3", "C4", "C5", "C6")

# Algoritmos que podem ser selecionados.
ALGORITHMS = ("minimax", "gpn_mcts")

# Orçamentos oficiais.
BUDGETS = (
    "100ms",
    "300ms",
    "1000ms",
    "500_transitions",
    "2000_transitions",
    "5000_transitions",
)


def normalize_algorithm(algorithm: str) -> str:
    """
    Normaliza o nome do algoritmo.
    """
    name = algorithm.strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "minimax": "minimax",
        "alphabeta": "minimax",
        "alpha_beta": "minimax",
        "gpn_mcts": "gpn_mcts",
        "gpnmcts": "gpn_mcts",
    }

    if name not in aliases:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")

    return aliases[name]


def normalize_budget_key(value: str) -> str:
    """
    Valida e normaliza o identificador do orçamento.
    """
    key = str(value).strip().lower().replace(" ", "")

    key = key.replace("-", "_")

    if key not in BUDGETS:
        raise ValueError(f"Orçamento inválido: {value}")

    return key


def budget_key_from_budget(budget: Budget) -> str:
    """
    Identifica qual dos seis orçamentos está sendo utilizado.

    Compatível com o contrato atual de algorithms/base.py:
        max_time_s
        max_transitions
    """
    if budget.max_time_s is not None and budget.max_transitions is not None:
        raise ValueError(
            "Utilize apenas orçamento de tempo ou de transições por execução."
        )

    if budget.max_time_s is not None:
        if budget.max_time_s <= 0:
            raise ValueError("O orçamento de tempo deve ser positivo.")

        milliseconds = round(budget.max_time_s * 1000)

        return normalize_budget_key(f"{milliseconds}ms")

    if budget.max_transitions is not None:
        if budget.max_transitions <= 0:
            raise ValueError("O orçamento de transições deve ser positivo.")

        return normalize_budget_key(
            f"{budget.max_transitions}_transitions"
        )

    raise ValueError("Nenhum orçamento foi informado.")


def laplace_probability(
    correct: float,
    attempts: int,
) -> float:
    """
    Calcula a probabilidade utilizando suavização de Laplace.

    P(Q=1) = (acertos + 1) / (tentativas + 2)

    'correct' pode ser fracionário porque as seeds do GPN-MCTS
    são primeiro agregadas por posição.
    """
    if attempts < 0:
        raise ValueError("O número de tentativas não pode ser negativo.")

    if not 0 <= correct <= attempts:
        raise ValueError("Quantidade de acertos inválida.")

    return (correct + 1) / (attempts + 2)


def expected_utility(probability: float) -> float:
    """
    Calcula a utilidade esperada da escolha.

    Utilidade:
        1 = coincidência com Stockfish
        0 = não coincidência

    EU = P(Q=1)*1 + P(Q=0)*0
    """
    if not 0 <= probability <= 1:
        raise ValueError("A probabilidade deve estar entre 0 e 1.")

    return probability


def parse_correct(value) -> float:
    """
    Converte o resultado de uma avaliação para 0 ou 1.
    """
    if isinstance(value, bool):
        return float(value)

    if str(value).strip().lower() in ("true", "yes"):
        return 1.0

    if str(value).strip().lower() in ("false", "no"):
        return 0.0

    result = float(value)

    if not 0 <= result <= 1:
        raise ValueError(f"Resultado inválido: {value}")

    return result


def build_probability_table(records) -> dict:
    """
    Constrói a tabela de probabilidades da calibração.

    Cada registro deve possuir:
        scenario_id
        context
        budget
        algorithm
        correct

    Resultados repetidos da mesma posição são primeiro
    agregados. Assim, as três seeds do GPN-MCTS não são
    tratadas como três posições independentes.

    Retorno:
        tabela[orçamento][contexto][algoritmo] = probabilidade
    """
    grouped = defaultdict(list)

    for record in records:
        scenario = str(record["scenario_id"])
        context = str(record["context"]).strip().upper()
        budget = normalize_budget_key(record["budget"])
        algorithm = normalize_algorithm(record["algorithm"])
        correct = parse_correct(record["correct"])

        if context not in CONTEXTS:
            raise ValueError(f"Contexto inválido: {context}")

        grouped[
            (budget, context, algorithm, scenario)
        ].append(correct)

    # Média de acertos por posição.
    position_results = defaultdict(list)

    for (budget, context, algorithm, scenario), values in grouped.items():
        position_accuracy = sum(values) / len(values)

        position_results[
            (budget, context, algorithm)
        ].append(position_accuracy)

    table = {}

    for (budget, context, algorithm), accuracies in position_results.items():

        correct = sum(accuracies)
        attempts = len(accuracies)

        probability = laplace_probability(correct, attempts)

        table.setdefault(budget, {})
        table[budget].setdefault(context, {})

        table[budget][context][algorithm] = probability

    return table


def load_calibration_csv(path: str | Path) -> dict:
    """
    Carrega os resultados de calibração de um CSV
    e constrói a tabela de probabilidades.
    """
    with open(path, newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    return build_probability_table(records)


def select_algorithm(probabilities: dict) -> str:
    """
    Seleciona o algoritmo com maior utilidade esperada.

    Em caso de empate, escolhe Minimax.
    """
    for algorithm in ALGORITHMS:
        if algorithm not in probabilities:
            raise KeyError(
                f"Probabilidade ausente para {algorithm}."
            )

    minimax_eu = expected_utility(
        probabilities["minimax"]
    )

    gpn_mcts_eu = expected_utility(
        probabilities["gpn_mcts"]
    )

    if gpn_mcts_eu > minimax_eu:
        return "gpn_mcts"

    return "minimax"


def select_fixed_algorithm(
    probability_table: dict,
    budget: str,
) -> str:
    """
    Implementa a política fixa de referência.

    Escolhe o algoritmo com maior concordância média
    na calibração, independentemente do contexto.

    Em caso de empate, escolhe Minimax.
    """
    budget = normalize_budget_key(budget)

    scores = {}

    for algorithm in ALGORITHMS:
        context_scores = []

        for context in CONTEXTS:
            try:
                probability = probability_table[
                    budget
                ][context][algorithm]
            except KeyError:
                raise KeyError(
                    f"Calibração incompleta: {budget}, "
                    f"{context}, {algorithm}."
                )

            context_scores.append(probability)

        scores[algorithm] = (
            sum(context_scores) / len(context_scores)
        )

    if scores["gpn_mcts"] > scores["minimax"]:
        return "gpn_mcts"

    return "minimax"


class HybridAgent(Agent):
    """
    Agente híbrido baseado em Máxima Utilidade Esperada.

    Recebe:
        - agente Minimax;
        - agente GPN-MCTS;
        - tabela de probabilidades da calibração.

    Executa apenas um algoritmo por decisão.
    """

    name = "hybrid"

    def __init__(
        self,
        minimax_agent: Agent,
        gpn_mcts_agent: Agent,
        probability_table: dict,
    ):
        self.agents = {
            "minimax": minimax_agent,
            "gpn_mcts": gpn_mcts_agent,
        }

        self.probability_table = probability_table

    def choose_action(
        self,
        state,
        budget: Budget,
    ) -> Decision:
        """
        Identifica o contexto, seleciona o algoritmo e executa
        a busca com o orçamento restante.
        """
        start = perf_counter()
        transitions_before = transition_count()

        # Obtém as ações legais uma única vez.
        actions = legal_actions(state)

        if not actions:
            raise ValueError(
                "Não é possível escolher uma ação em estado terminal."
            )

        # Extrai as características da posição.
        features = state_features(state, actions)

        # Identifica o orçamento.
        budget_key = budget_key_from_budget(budget)

        # Consulta as probabilidades da calibração.
        try:
            probabilities = self.probability_table[
                budget_key
            ][features["context"]]
        except KeyError as error:
            raise KeyError(
                f"Calibração ausente para orçamento "
                f"{budget_key} e contexto {features['context']}."
            ) from error

        # Seleciona o algoritmo pela MEU.
        selected_algorithm = select_algorithm(probabilities)

        # Calcula o custo de extração e seleção.
        selection_time = perf_counter() - start

        selection_transitions = (
            transition_count() - transitions_before
        )

        # Desconta o custo de seleção do orçamento.
        remaining_time = None

        if budget.max_time_s is not None:
            remaining_time = max(
                0.0,
                budget.max_time_s - selection_time
            )

        remaining_transitions = None

        if budget.max_transitions is not None:
            remaining_transitions = max(
                0,
                budget.max_transitions - selection_transitions
            )

        hybrid_info = {
            "context": features["context"],
            "features": features,
            "budget": budget_key,
            "probabilities": dict(probabilities),
            "expected_utilities": {
                algorithm: expected_utility(
                    probabilities[algorithm]
                )
                for algorithm in ALGORITHMS
            },
            "selected_algorithm": selected_algorithm,
            "selection_time_ms": selection_time * 1000,
            "selection_transitions": selection_transitions,
        }

        # Se a seleção consumir o orçamento, retorna
        # uma ação legal determinística como fallback.
        if (
            remaining_time is not None and remaining_time <= 0
        ) or (
            remaining_transitions is not None
            and remaining_transitions <= 0
        ):
            hybrid_info["fallback"] = "budget_exhausted"

            return Decision(
                action=actions[0],
                transitions=selection_transitions,
                extra={"hybrid": hybrid_info},
            )

        # Cria o orçamento restante para o algoritmo escolhido.
        remaining_budget = Budget(
            max_time_s=remaining_time,
            max_transitions=remaining_transitions,
            max_depth=budget.max_depth,
        )

        # Executa somente o algoritmo selecionado.
        decision = self.agents[selected_algorithm].choose_action(
            state,
            remaining_budget,
        )

        # Verifica se a ação retornada é legal.
        if decision.action not in actions:
            raise ValueError(
                f"{selected_algorithm} retornou uma ação ilegal: "
                f"{decision.action}"
            )

        # Registra o custo total da decisão.
        total_transitions = (
            transition_count() - transitions_before
        )

        hybrid_info["total_transitions"] = total_transitions
        hybrid_info["total_time_ms"] = (
            perf_counter() - start
        ) * 1000

        return Decision(
            action=decision.action,
            transitions=total_transitions,
            nodes_expanded=decision.nodes_expanded,
            nodes_visited=decision.nodes_visited,
            max_depth_reached=decision.max_depth_reached,
            extra={
                "agent_extra": decision.extra,
                "hybrid": hybrid_info,
            },
        )
