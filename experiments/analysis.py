"""
analysis.py - Tabelas e graficos do artigo (Plano v2.0, secoes 10.3, 11.6 e 13).

Entradas (experiments/results/): decisions.csv, quality.csv, games.csv
Saidas:
  results/calibration_records.csv -> ENTREGAR A CLARA (formato do load_calibration_csv dela)
  results/calibration_table.csv  -> mesma calibracao em contagens (conferencia)
  results/empirical_summary.csv  -> analise empirica de complexidade
  plots/01..08_*.png             -> os 8 graficos obrigatorios

Uso:  python -m experiments.analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAIN_TIME_MS = 300
AGENT_ORDER = ["minimax", "gpn_mcts", "hybrid", "fixed_policy"]
AGENT_LABEL = {"minimax": "Minimax + Alfa-Beta", "gpn_mcts": "GPN-MCTS",
               "hybrid": "Hibrida (MEU)", "fixed_policy": "Politica fixa"}
BUDGET_LABEL = {"time": "Orcamento de tempo (ms por decisao)",
                "transitions": "Orcamento de transicoes por decisao"}


# ----------------------------------------------------------------------------
# Carregamento e tabelas
# ----------------------------------------------------------------------------
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def load_all(results_dir: str | Path):
    d = Path(results_dir)
    return _read(d / "decisions.csv"), _read(d / "quality.csv"), _read(d / "games.csv")


def merged_quality(decisions: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """Junta as decisoes de TEMPO com a qualidade (so decisoes sem falha)."""
    if decisions.empty or quality.empty:
        return pd.DataFrame()
    dec = decisions[(decisions.run_kind == "time") & (decisions.status == "ok")]
    q = quality[["experiment_id", "run_id", "top1", "regret_cp", "decisive_class"]]
    return dec.merge(q, on=["experiment_id", "run_id"], how="inner")


def per_position(mq: pd.DataFrame) -> pd.DataFrame:
    """Media POR POSICAO (seeds e repeticoes viram uma unica observacao por posicao).

    As tres seeds NAO sao tres posicoes independentes (plano, secao 10.3).
    """
    if mq.empty:
        return mq
    mq = mq.assign(top1=mq.top1.astype(float))
    keys = ["position_set", "context", "position_id", "agent", "budget_type", "budget_value"]
    return mq.groupby(keys, as_index=False).agg(top1=("top1", "mean"),
                                                regret_cp=("regret_cp", "mean"))


def calibration_table(pp: pd.DataFrame) -> pd.DataFrame:
    """Contagens para a Clara aplicar a suavizacao de Laplace:
        P(Q=1 | a, x, B) = (acertos + 1) / (tentativas + 2)

    tentativas = numero de posicoes; acertos = soma das proporcoes de acerto por posicao
    (fracionario no GPN-MCTS, pois cada posicao tem 3 seeds com o mesmo peso).
    """
    cal = pp[(pp.position_set == "calibration") & pp.agent.isin(["minimax", "gpn_mcts"])]
    if cal.empty:
        return pd.DataFrame(columns=["agent", "context", "budget_type", "budget_value",
                                     "tentativas", "acertos"])
    t = cal.groupby(["agent", "context", "budget_type", "budget_value"], as_index=False).agg(
        tentativas=("top1", "size"), acertos=("top1", "sum"))
    return t.sort_values(["budget_type", "budget_value", "context", "agent"]).reset_index(drop=True)


def budget_key(budget_type: str, value: float) -> str:
    """Identificador de orcamento usado pela hibrida: '300ms' ou '2000_transitions'."""
    return f"{int(round(value))}ms" if budget_type == "time" else f"{int(round(value))}_transitions"


def calibration_records(mq: pd.DataFrame) -> pd.DataFrame:
    """Registros de calibracao no formato que `algorithms.hybrid.load_calibration_csv` le.

    Uma linha por DECISAO (cada seed e cada repeticao); a hibrida agrega por posicao.
    Colunas: scenario_id, context, budget, algorithm, correct (1 = top-1 com o Stockfish).
    """
    cols = ["scenario_id", "context", "budget", "algorithm", "correct"]
    if mq.empty:
        return pd.DataFrame(columns=cols)
    c = mq[(mq.position_set == "calibration") & mq.agent.isin(["minimax", "gpn_mcts"])]
    out = pd.DataFrame({
        "scenario_id": c.position_id,
        "context": c.context,
        "budget": [budget_key(t, v) for t, v in zip(c.budget_type, c.budget_value)],
        "algorithm": c.agent,
        "correct": c.top1.astype(int),
    })
    return out.sort_values(["budget", "context", "algorithm", "scenario_id"]).reset_index(drop=True)


def derive_fixed_policy(pp: pd.DataFrame) -> pd.DataFrame:
    """Politica fixa (secao 10.5): por orcamento, o algoritmo com maior concordancia media
    na CALIBRACAO (empate -> Minimax), aplicado ao conjunto de TESTE sem olhar o contexto."""
    cal = pp[(pp.position_set == "calibration") & pp.agent.isin(["minimax", "gpn_mcts"])]
    test = pp[pp.position_set == "test"]
    rows = []
    for (btype, bval), g in cal.groupby(["budget_type", "budget_value"]):
        means = g.groupby("agent").top1.mean()
        mm, gp = means.get("minimax", -1), means.get("gpn_mcts", -1)
        chosen = "gpn_mcts" if gp > mm else "minimax"
        sel = test[(test.agent == chosen) & (test.budget_type == btype) & (test.budget_value == bval)]
        rows.append(sel.assign(agent="fixed_policy", fixed_choice=chosen))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def empirical_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    """Medias por agente e orcamento (rodada de TEMPO, sem falhas) + ramificacao efetiva aprox."""
    if decisions.empty:
        return pd.DataFrame()
    d = decisions[(decisions.run_kind == "time") & (decisions.status == "ok")].copy()
    cols = [c for c in ["nodes_visited", "nodes_expanded", "tree_nodes_created", "transitions",
                        "max_depth_reached", "elapsed_ms", "cpu_ms", "alpha_beta_prunes",
                        "complete_depth", "iterations", "playouts", "proof_updates",
                        "pnrank_sorts", "selection_ms"] if c in d.columns]
    s = d.groupby(["agent", "budget_type", "budget_value"], as_index=False)[cols].mean()
    mm = d[(d.agent == "minimax") & (d.get("complete_depth", pd.Series(dtype=float)) > 0)]
    if not mm.empty:
        b = (mm.nodes_visited ** (1.0 / mm.complete_depth)).groupby(
            [mm.budget_type, mm.budget_value]).mean().rename("b_efetivo_aprox").reset_index()
        b["agent"] = "minimax"
        s = s.merge(b, on=["agent", "budget_type", "budget_value"], how="left")
    return s


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------
def _save(fig, plots_dir: Path, name: str) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _agents_in(df) -> list[str]:
    return [a for a in AGENT_ORDER if a in set(df.agent)]


def plot_quality_vs_budget(pp, plots_dir):
    """Grafico 1: qualidade x orcamento (top-1 e perda em centipeoes), conjunto de teste."""
    out = []
    test = pp[pp.position_set == "test"] if not pp.empty else pp
    for btype in (test.budget_type.unique() if not test.empty else []):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, col, ylab in ((axes[0], "top1", "Concordancia top-1 com o Stockfish"),
                              (axes[1], "regret_cp", "Perda media (centipeoes)")):
            for ag in _agents_in(test):
                g = test[(test.agent == ag) & (test.budget_type == btype)]
                s = g.groupby("budget_value")[col].agg(["mean", "std", "count"]).reset_index()
                if s.empty:
                    continue
                se = (s["std"].fillna(0) / np.sqrt(s["count"])).values
                ax.errorbar(s.budget_value, s["mean"], yerr=se, marker="o", capsize=3,
                            label=AGENT_LABEL[ag])
            ax.set_xscale("log")
            ax.set_xlabel(BUDGET_LABEL[btype])
            ax.set_ylabel(ylab)
            ax.grid(alpha=.3)
        axes[0].legend()
        out.append(_save(fig, plots_dir, f"01_qualidade_orcamento_{btype}.png"))
    return out


def plot_time_vs_transitions(dec, plots_dir):
    """Grafico 2: tempo real x transicoes."""
    d = dec[(dec.run_kind == "time") & (dec.status == "ok")] if not dec.empty else dec
    if d.empty:
        return []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ag in _agents_in(d):
        g = d[(d.agent == ag) & (d.transitions > 0)]
        ax.scatter(g.transitions, g.elapsed_ms, s=8, alpha=.4, label=AGENT_LABEL[ag])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Transicoes por decisao")
    ax.set_ylabel("Tempo real (ms)")
    ax.grid(alpha=.3)
    ax.legend()
    return [_save(fig, plots_dir, "02_tempo_transicoes.png")]


def _main_budget(d):
    m = d[(d.budget_type == "time") & (d.budget_value == MAIN_TIME_MS)]
    return (m, f"{MAIN_TIME_MS} ms") if not m.empty else (d, "todos os orcamentos")


def plot_nodes(dec, plots_dir):
    """Grafico 3: nos visitados e expandidos por algoritmo."""
    d = dec[(dec.run_kind == "time") & (dec.status == "ok")] if not dec.empty else dec
    if d.empty:
        return []
    d, label = _main_budget(d)
    ags = _agents_in(d)
    m = d.groupby("agent")[["nodes_visited", "nodes_expanded"]].mean().reindex(ags)
    x = np.arange(len(ags))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - .2, m.nodes_visited, .4, label="Visitados")
    ax.bar(x + .2, m.nodes_expanded, .4, label="Expandidos")
    ax.set_xticks(x, [AGENT_LABEL[a] for a in ags])
    ax.set_ylabel("Nos por decisao (media)")
    ax.set_title(f"Orcamento: {label}")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    return [_save(fig, plots_dir, "03_nos_visitados_expandidos.png")]


def plot_depth(dec, plots_dir):
    """Grafico 4: profundidade atingida (meios-lances)."""
    d = dec[(dec.run_kind == "time") & (dec.status == "ok")] if not dec.empty else dec
    out = []
    for btype in (d.budget_type.unique() if not d.empty else []):
        fig, ax = plt.subplots(figsize=(6, 4))
        for ag in _agents_in(d):
            g = d[(d.agent == ag) & (d.budget_type == btype)]
            s = g.groupby("budget_value").max_depth_reached.mean()
            ax.plot(s.index, s.values, marker="o", label=AGENT_LABEL[ag])
        ax.set_xscale("log")
        ax.set_xlabel(BUDGET_LABEL[btype])
        ax.set_ylabel("Profundidade maxima (meios-lances, media)")
        ax.grid(alpha=.3)
        ax.legend()
        out.append(_save(fig, plots_dir, f"04_profundidade_{btype}.png"))
    return out


def plot_memory(dec, plots_dir):
    """Grafico 5: pico de memoria (rodada separada com tracemalloc)."""
    m = dec[(dec.run_kind == "memory") & (dec.status == "ok")] if not dec.empty else dec
    if m.empty:
        return []
    out = []
    for btype in m.budget_type.unique():
        g = m[m.budget_type == btype]
        piv = g.groupby(["budget_value", "agent"]).peak_memory_kb.mean().unstack("agent")
        piv = piv[[a for a in AGENT_ORDER if a in piv.columns]]
        fig, ax = plt.subplots(figsize=(6, 4))
        piv.rename(columns=AGENT_LABEL).plot.bar(ax=ax, rot=0)
        ax.set_xlabel(BUDGET_LABEL[btype])
        ax.set_ylabel("Pico de memoria (KB)")
        ax.grid(axis="y", alpha=.3)
        out.append(_save(fig, plots_dir, f"05_memoria_{btype}.png"))
    return out


def plot_quality_by_context(pp, plots_dir):
    """Grafico 6: qualidade por contexto (C1..C6), conjunto de teste."""
    test = pp[pp.position_set == "test"] if not pp.empty else pp
    if test.empty:
        return []
    sel = test[(test.budget_type == "time") & (test.budget_value == MAIN_TIME_MS)]
    label = f"{MAIN_TIME_MS} ms"
    if sel.empty:
        first = test.groupby(["budget_type", "budget_value"]).ngroups and test.iloc[0]
        sel = test[(test.budget_type == first.budget_type) & (test.budget_value == first.budget_value)]
        label = f"{first.budget_type}={first.budget_value:g}"
    piv = sel.groupby(["context", "agent"]).top1.mean().unstack("agent")
    piv = piv[[a for a in AGENT_ORDER if a in piv.columns]]
    fig, ax = plt.subplots(figsize=(8, 4))
    piv.rename(columns=AGENT_LABEL).plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Concordancia top-1")
    ax.set_xlabel("Contexto (C1 abertura ... C6 final tatico)")
    ax.set_title(f"Orcamento: {label}")
    ax.grid(axis="y", alpha=.3)
    return [_save(fig, plots_dir, "06_qualidade_contexto.png")]


def plot_gpn_cost_per_iteration(dec, plots_dir):
    """Grafico 7: custo por iteracao do GPN-MCTS (ms por iteracao)."""
    if dec.empty or "iterations" not in dec.columns:
        return []
    g = dec[(dec.run_kind == "time") & (dec.agent == "gpn_mcts") & (dec.status == "ok")
            & (dec.iterations > 0)].copy()
    if g.empty:
        return []
    g["ms_por_iteracao"] = g.elapsed_ms / g.iterations
    fig, ax = plt.subplots(figsize=(6, 4))
    for btype in g.budget_type.unique():
        s = g[g.budget_type == btype].groupby("budget_value").ms_por_iteracao.agg(["mean", "std"])
        ax.errorbar(range(len(s)), s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3,
                    label=btype)
        ax.set_xticks(range(len(s)), [f"{v:g}" for v in s.index])
    ax.set_xlabel("Orcamento")
    ax.set_ylabel("Tempo por iteracao (ms)")
    ax.grid(alpha=.3)
    ax.legend()
    return [_save(fig, plots_dir, "07_custo_iteracao_gpn.png")]


def plot_games(games, plots_dir):
    """Grafico 8: resultado das partidas completas (vitorias, empates, derrotas)."""
    if games.empty:
        return []
    done = games[games.status == "completed"]
    rows = []
    for matchup, g in done.groupby("matchup"):
        a, b = matchup.split("_vs_")
        wa = ((g.white == a) & (g.result == "1-0")).sum() + ((g.black == a) & (g.result == "0-1")).sum()
        wb = ((g.white == b) & (g.result == "1-0")).sum() + ((g.black == b) & (g.result == "0-1")).sum()
        rows.append((f"{AGENT_LABEL[a]}\nx {AGENT_LABEL[b]}", wa, (g.result == "1/2-1/2").sum(), wb))
    if not rows:
        return []
    labels, wa, dr, wb = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 3 + .6 * len(rows)))
    y = np.arange(len(rows))
    ax.barh(y, wa, label="Vitorias do 1o agente")
    ax.barh(y, dr, left=wa, label="Empates")
    ax.barh(y, wb, left=np.array(wa) + np.array(dr), label="Vitorias do 2o agente")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Partidas (de 12 por confronto)")
    ax.legend(loc="lower right")
    return [_save(fig, plots_dir, "08_partidas.png")]


# ----------------------------------------------------------------------------
def run_all(results_dir="experiments/results", plots_dir="experiments/plots") -> None:
    results_dir, plots_dir = Path(results_dir), Path(plots_dir)
    dec, qual, games = load_all(results_dir)
    mq = merged_quality(dec, qual)
    pp = per_position(mq)
    if not pp.empty:
        calibration_records(mq).to_csv(results_dir / "calibration_records.csv", index=False)
        calibration_table(pp).to_csv(results_dir / "calibration_table.csv", index=False)
        fixed = derive_fixed_policy(pp)
        pp = pd.concat([pp, fixed], ignore_index=True) if not fixed.empty else pp
        pp.to_csv(results_dir / "quality_per_position.csv", index=False)
    summary = empirical_summary(dec)
    if not summary.empty:
        summary.to_csv(results_dir / "empirical_summary.csv", index=False)
    made = []
    made += plot_quality_vs_budget(pp, plots_dir) if not pp.empty else []
    made += plot_time_vs_transitions(dec, plots_dir)
    made += plot_nodes(dec, plots_dir)
    made += plot_depth(dec, plots_dir)
    made += plot_memory(dec, plots_dir)
    made += plot_quality_by_context(pp, plots_dir) if not pp.empty else []
    made += plot_gpn_cost_per_iteration(dec, plots_dir)
    made += plot_games(games, plots_dir)
    print(f"{len(made)} graficos gerados em {plots_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--plots", default="experiments/plots")
    a = ap.parse_args()
    run_all(a.results, a.plots)


if __name__ == "__main__":
    main()
