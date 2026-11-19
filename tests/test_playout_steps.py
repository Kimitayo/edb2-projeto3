import pytest

from algorithms.base import Budget
from algorithms.gpn_mcts import GPNMCTSAgent
from algorithms.mcts import MCTSAgent
from game import initial_state


@pytest.mark.parametrize("classe_agente", [MCTSAgent, GPNMCTSAgent])
def test_conta_passo_de_playout_interrompido(classe_agente):
    agente = classe_agente(seed=2026)
    decisao = agente.choose_action(initial_state(), Budget(max_transitions=2))

    # Uma transição cria o filho e a outra executa um passo da simulação.
    assert decisao.transitions == 2
    assert decisao.extra["playout_steps"] == 1
    assert decisao.extra["playouts"] == 0