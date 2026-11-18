import pandas as pd

from experiments import analysis


def test_grafico_agrupa_partidas_com_cores_invertidas(tmp_path, monkeypatch):
    confrontos = [
        ("minimax", "gpn_mcts"),
        ("hybrid", "minimax"),
        ("hybrid", "gpn_mcts"),
    ]

    registros = []
    for a, b in confrontos:
        # Com A de brancas: 4 vitórias de A, 1 empate e 1 vitória de B.
        # Com A de pretas:  3 vitórias de A, 2 empates e 1 vitória de B.
        for brancas, pretas, resultados in [
            (a, b, ["1-0"] * 4 + ["1/2-1/2", "0-1"]),
            (b, a, ["0-1"] * 3 + ["1/2-1/2"] * 2 + ["1-0"]),
        ]:
            for resultado in resultados:
                registros.append({
                    "white": brancas,
                    "black": pretas,
                    "matchup": f"{brancas}_vs_{pretas}",
                    "result": resultado,
                    "status": "completed",
                })

    # Lê os valores das barras no momento em que o gráfico seria salvo.
    contagens = {}

    def conferir_barras(fig, pasta, nome):
        eixo = fig.axes[0]
        rotulos = [rotulo.get_text() for rotulo in eixo.get_yticklabels()]
        vitorias_a, empates, vitorias_b = eixo.containers

        for rotulo, va, em, vb in zip(
            rotulos, vitorias_a, empates, vitorias_b, strict=True
        ):
            contagens[rotulo] = (va.get_width(), em.get_width(), vb.get_width())

        analysis.plt.close(fig)
        return pasta / nome

    monkeypatch.setattr(analysis, "_save", conferir_barras)
    analysis.plot_games(pd.DataFrame(registros), tmp_path)

    esperado = {
        f"{analysis.AGENT_LABEL[a]}\nx {analysis.AGENT_LABEL[b]}": (7, 3, 2)
        for a, b in confrontos
    }
    assert contagens == esperado