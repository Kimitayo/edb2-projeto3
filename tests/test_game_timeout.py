from contextlib import nullcontext

import chess

from algorithms.base import Decision
from experiments import runner
from experiments.metrics import Measurement
from experiments.results_storage import decision_table, load_rows


def test_timeout_encerra_partida_sem_pontuar(tmp_path, monkeypatch):
    # Simula 601 ms de execução sem depender da velocidade do computador.
    medida = Measurement(elapsed_ms=601)
    monkeypatch.setattr(runner, "measure", lambda _: nullcontext(medida))

    class AgenteDeTeste:
        def choose_action(self, board, budget):
            return Decision(action=chess.Move.from_uci("e2e4"))

    agentes = {
        "minimax": lambda seed: AgenteDeTeste(),
        "gpn_mcts": lambda seed: AgenteDeTeste(),
    }

    caminho_csv = tmp_path / "decisions.csv"
    with decision_table(caminho_csv) as decisoes:
        partida = runner.play_game(
            white="minimax",
            black="gpn_mcts",
            budget=runner.make_budget("time", 300),
            game_id="teste_timeout",
            seeds=(None, 2026),
            factories=agentes,
            decisions=decisoes,
            experiment_id="teste",
            commit="teste",
            pgn_dir=tmp_path,
            max_plies=1,
        )

    # O CSV preserva a tentativa e identifica o timeout.
    registros = load_rows(caminho_csv)
    assert len(registros) == 1
    assert registros[0]["action_uci"] == "e2e4"
    assert registros[0]["status"] == "timeout"

    # A partida termina antes de aplicar o lance e sem atribuir pontos.
    assert partida.plies == 0
    assert partida.status == "failure"
    assert partida.termination == "timeout:minimax"
    assert partida.failures == 1
    assert partida.result == "*"
    assert partida.points_white is None
    assert partida.points_black is None