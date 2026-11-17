"""
runner.py - Executa os experimentos (Plano v2.0, secoes 8, 9.5 e 14).

O runner so MEDE: ele chama `agent.choose_action(state, budget)` (contrato da
secao 12.2), cronometra de fora e grava uma linha por decisao. Nenhum agente
consulta o Stockfish; a qualidade e avaliada depois, por experiments.reference.

Rodadas separadas (secao 8.3):
  --kind time    : mede tempo real e CPU (sem tracemalloc)
  --kind memory  : mede o pico de memoria (tracemalloc); o tempo desta rodada NAO vale

Exemplos:
  python -m experiments.runner decisions --set calibration --budget-type time
  python -m experiments.runner decisions --set test --budget-type time --agents minimax,gpn_mcts,hybrid \
         --calibration-records experiments/results/calibration_records.csv
  python -m experiments.runner decisions --set test --budget-type time --kind memory
  python -m experiments.runner games --calibration-records experiments/results/calibration_records.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from experiments.metrics import (Measurement, budget_overrun, budget_parts, is_timeout, measure)
from experiments.results_storage import (DecisionRecord, GameRecord, decision_table, game_table,
                                         load_rows, next_run_id)

try:  # contrato oficial (secao 12.2)
    from algorithms.base import Budget
except ImportError:  # so para desenvolver ANTES de algorithms/base.py existir
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class Budget:  # mesmo formato do algorithms/base.py atual
        max_transitions: int | None = None
        max_time_s: float | None = None
        max_depth: int | None = None

SEEDS = (2026, 2027, 2028)
SEEDED_AGENTS = ("gpn_mcts", "hybrid")
TIME_BUDGETS_MS = (100, 300, 1000)
TRANSITION_BUDGETS = (500, 2000, 5000)
GAME_BUDGET_MS = 300
GAMES_PER_MATCHUP = 12
MATCHUPS = (("minimax", "gpn_mcts"), ("hybrid", "minimax"), ("hybrid", "gpn_mcts"))
MAX_PLIES = 500          # trava de seguranca (nao definida no plano); partida vira "max_plies"
DEFAULT_REPS = 3         # repeticoes de MEDICAO de tempo (o plano nao fixa o numero)

# Chaves esperadas em `Decision.extra` (combinar os nomes com Flavio, Gustavo e Clara).
EXTRA_FIELDS = ("alpha_beta_prunes", "complete_depth", "iterations", "playouts", "playout_steps",
                "proof_updates", "pnrank_sorts", "chosen_algorithm", "selection_ms",
                "p_minimax", "p_gpn_mcts", "tree_nodes_created")


# Nomes que os colegas usaram -> nomes das colunas do CSV
EXTRA_ALIASES = {"pn_updates": "proof_updates", "pn_rank_sorts": "pnrank_sorts",
                 "tree_nodes": "tree_nodes_created"}
# Medidas que o agente relata sobre si mesmo. O runner mede de fora, e a medida DELE e a
# valida; o que o agente relatou fica guardado em extra_json["agent_reported"].
SELF_REPORTED = ("elapsed_ms", "cpu_ms", "peak_memory_kb", "budget_overrun", "status")


# ----------------------------------------------------------------------------
# Agentes (cada colega entregou de um jeito: funcao ou classe)
# ----------------------------------------------------------------------------
class FunctionAgent:
    """Adapta uma funcao `choose_action(state, budget, *, seed=None, ...)` ao contrato de objeto.

    So repassa os argumentos que a funcao realmente aceita (seed, table_path...).
    """

    def __init__(self, fn, **options):
        self.fn = fn
        params = inspect.signature(fn).parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        self.kwargs = {k: v for k, v in options.items()
                       if v is not None and (takes_any or k in params)}

    def choose_action(self, state, budget):
        return self.fn(state, budget, **self.kwargs)


def _load_agent(module: str, class_names: tuple[str, ...], **options):
    """Cria o agente de `module`.

    Ordem: (1) a primeira classe de `class_names` que existir; (2) a unica classe do modulo
    que tenha `choose_action`; (3) a funcao `choose_action` do modulo. So repassa os argumentos
    que o construtor/funcao aceita (seed, ...).
    """
    mod = importlib.import_module(module)
    cls = next((getattr(mod, n) for n in class_names if hasattr(mod, n)), None)
    if cls is None:
        own = [c for c in vars(mod).values()
               if inspect.isclass(c) and c.__module__ == module and hasattr(c, "choose_action")]
        cls = own[0] if len(own) == 1 else None
    if cls is not None:
        params = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in options.items() if v is not None and k in params})
    return FunctionAgent(mod.choose_action, **options)


def make_factories(calibration_records: str | None = None) -> dict:
    """nome -> funcao(seed) que cria um agente NOVO (sem estado de decisoes anteriores).

    GPN-MCTS (Gustavo): classe `GPNMCTSAgent(seed=...)` em algorithms/gpn_mcts.py.
    Hibrida (Clara): `HybridAgent(minimax_agent, gpn_mcts_agent, probability_table)`; a tabela
    sai de `load_calibration_csv(calibration_records.csv)`, arquivo gerado por experiments.analysis.
    Minimax (Flavio): ainda nao entregue; tenta `MinimaxAgent`/`Minimax`/`AlphaBetaAgent`, a unica
    classe com `choose_action` do modulo, ou uma funcao `choose_action`. Ajustar quando ele entregar.
    """
    def minimax(seed):
        return _load_agent("algorithms.minimax", ("MinimaxAgent", "Minimax", "AlphaBetaAgent"))

    def gpn_mcts(seed):
        return _load_agent("algorithms.gpn_mcts", ("GPNMCTSAgent", "GPNMCTS"), seed=seed)

    def hybrid(seed):
        from algorithms.hybrid import HybridAgent, load_calibration_csv
        if not calibration_records:
            raise SystemExit("A hibrida precisa de --calibration-records (CSV de calibracao).")
        table = load_calibration_csv(calibration_records)
        return HybridAgent(minimax(seed), gpn_mcts(seed), table)

    return {"minimax": minimax, "gpn_mcts": gpn_mcts, "hybrid": hybrid}


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# Uma decisao
# ----------------------------------------------------------------------------
@dataclass
class Outcome:
    decision: object | None
    measurement: Measurement
    status: str
    error: str
    overrun: bool
    transitions: int | None = None


def _is_claim_draw(action) -> bool:
    return type(action).__name__ == "ClaimDraw"


def _is_legal(board, action) -> bool:
    """A acao esta entre as acoes legais do MOTOR (inclui ClaimDraw)?"""
    try:
        from game import legal_actions
    except ImportError:          # sem motor (testes com tabuleiro falso)
        if _is_claim_draw(action):
            return bool(board.can_claim_draw())
        try:
            return action in board.legal_moves
        except Exception:
            return False
    try:
        return action in legal_actions(board)
    except Exception:
        return False


def _transition_counter():
    """(reset, leitura) do contador global de transicoes do motor, ou (None, None)."""
    try:
        from game import reset_transition_count, transition_count
        return reset_transition_count, transition_count
    except ImportError:
        return None, None


def action_to_str(action) -> str:
    if _is_claim_draw(action):
        return "CLAIM_DRAW"
    return action.uci() if hasattr(action, "uci") else str(action)


def run_decision(agent, board, budget, track_memory: bool = False) -> Outcome:
    """Cronometra UMA decisao e confere o contrato (legalidade, estado intacto, orcamento)."""
    fen_before, stack_before = board.fen(), len(board.move_stack)
    decision, status, error = None, "ok", ""
    reset_count, read_count = _transition_counter()
    if reset_count:
        reset_count()
    with measure(track_memory) as m:
        try:
            decision = agent.choose_action(board, budget)
        except Exception as e:  # falha do agente nunca vira empate
            status, error = "error", repr(e)
    # O Decision atual (algorithms/base.py) nao tem `transitions`: usa o contador do motor.
    transitions = getattr(decision, "transitions", None)
    if transitions is None and read_count:
        transitions = read_count()
    if status == "ok":
        if board.fen() != fen_before or len(board.move_stack) != stack_before:
            status, error = "error", "o estado recebido foi modificado pelo agente"
        elif not _is_legal(board, decision.action):
            status, error = "illegal_action", f"acao ilegal: {action_to_str(decision.action)}"
        elif is_timeout(budget, m.elapsed_ms):
            status, error = "timeout", f"{m.elapsed_ms:.0f} ms passou do limite de timeout"
    overrun = budget_overrun(budget, m.elapsed_ms, transitions)
    return Outcome(decision, m, status, error, overrun, transitions)


def build_record(out: Outcome, *, experiment_id, run_id, run_kind, position_id, position_set,
                 context, game_id, ply, agent, seed, repetition, budget, commit,
                 action_changed=None) -> DecisionRecord:
    btype, bvalue = budget_parts(budget)
    d = out.decision
    extra = dict(getattr(d, "extra", None) or {}) if d is not None else {}
    # Hibrida (Clara): {"hybrid": {...}, "agent_extra": {...do algoritmo escolhido...}}
    inner = extra.pop("agent_extra", None)
    hyb = extra.pop("hybrid", None)
    if isinstance(inner, dict):
        for k, v in inner.items():
            extra.setdefault(k, v)
    if isinstance(hyb, dict):
        probs = hyb.get("probabilities") or {}
        extra["chosen_algorithm"] = hyb.get("selected_algorithm")
        extra["context"] = hyb.get("context")
        extra["selection_ms"] = hyb.get("selection_time_ms")
        extra["p_minimax"] = probs.get("minimax")
        extra["p_gpn_mcts"] = probs.get("gpn_mcts")
        detail = {k: v for k, v in hyb.items()
                  if k not in ("probabilities", "selected_algorithm", "context", "selection_time_ms")}
        if detail:
            extra["hybrid_details"] = detail
    for old, new in EXTRA_ALIASES.items():
        if old in extra and new not in extra:
            extra[new] = extra.pop(old)
    reported = {k: extra.pop(k) for k in SELF_REPORTED if k in extra}
    if reported:
        extra["agent_reported"] = reported
    take = {k: extra.pop(k, None) for k in EXTRA_FIELDS}
    hybrid_context = extra.pop("context", "") or ""
    tree_nodes = take.pop("tree_nodes_created")
    chosen = take.pop("chosen_algorithm") or ""
    return DecisionRecord(
        experiment_id=experiment_id, run_id=run_id, run_kind=run_kind, position_id=position_id,
        position_set=position_set, context=context, game_id=game_id, ply=ply, agent=agent,
        seed=seed, repetition=repetition, budget_type=btype, budget_value=bvalue,
        git_commit=commit,
        action_uci=action_to_str(d.action) if d is not None else "",
        action_changed=action_changed,
        nodes_visited=getattr(d, "nodes_visited", None),
        nodes_expanded=getattr(d, "nodes_expanded", None),
        tree_nodes_created=tree_nodes,
        transitions=out.transitions,
        max_depth_reached=getattr(d, "max_depth_reached", None),
        elapsed_ms=out.measurement.elapsed_ms, cpu_ms=out.measurement.cpu_ms,
        peak_memory_kb=out.measurement.peak_memory_kb if run_kind == "memory" else None,
        budget_overrun=out.overrun, status=out.status,
        chosen_algorithm=chosen, hybrid_context=str(hybrid_context),
        error_message=out.error,
        extra_json=json.dumps(extra, ensure_ascii=False, default=str) if extra else "",
        **{k: v for k, v in take.items()
           if k not in ("tree_nodes_created", "chosen_algorithm")},
    )


# ----------------------------------------------------------------------------
# Experimento sobre as 120 posicoes
# ----------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):g}"
    except ValueError:
        return str(v)


def _key(kind, pos_id, agent, seed, rep, btype, bvalue) -> tuple:
    return (kind, pos_id, agent, _fmt(seed), _fmt(rep), btype, _fmt(bvalue))


def make_budget(budget_type: str, value: int) -> Budget:
    """Cria o Budget no formato que algorithms/base.py tiver (max_time_ms ou max_time_s)."""
    if budget_type != "time":
        return Budget(max_transitions=value)
    names = {f.name for f in dataclasses.fields(Budget)}
    return Budget(max_time_ms=value) if "max_time_ms" in names else Budget(max_time_s=value / 1000)


def run_positions(positions, agent_names, budget_type, budget_values, *, table, experiment_id,
                  run_kind="time", reps=DEFAULT_REPS, seeds=SEEDS, factories=None,
                  commit="unknown", verbose=True) -> int:
    """Roda todas as combinacoes e grava as linhas. Pode ser interrompido e retomado."""
    factories = factories or make_factories()
    reps = reps if run_kind == "time" else 1          # memoria: 1 execucao por combinacao
    done = {_key(r["run_kind"], r["position_id"], r["agent"], r["seed"], r["repetition"],
                 r["budget_type"], r["budget_value"]) for r in load_rows(table.path)}
    run_id = next_run_id(table.path)
    first_action: dict[tuple, str] = {}
    written = 0
    for pos in positions:
        for value in budget_values:
            budget = make_budget(budget_type, value)
            for name in agent_names:
                for seed in (seeds if name in SEEDED_AGENTS else (None,)):
                    for rep in range(reps):
                        k = _key(run_kind, pos.position_id, name, seed, rep, budget_type, value)
                        if k in done:
                            continue
                        board = pos.to_board()                    # estado novo, com historico
                        agent = factories[name](seed)             # agente novo: sem arvore reaproveitada
                        out = run_decision(agent, board, budget, track_memory=(run_kind == "memory"))
                        action = action_to_str(out.decision.action) if out.decision else ""
                        ck = (pos.position_id, name, _fmt(seed), _fmt(value))
                        changed = None
                        if rep == 0:
                            first_action[ck] = action
                        elif ck in first_action:
                            changed = action != first_action[ck]
                        table.write(build_record(
                            out, experiment_id=experiment_id, run_id=run_id, run_kind=run_kind,
                            position_id=pos.position_id, position_set=pos.set, context=pos.context,
                            game_id="", ply=None, agent=name, seed=seed, repetition=rep,
                            budget=budget, commit=commit, action_changed=changed))
                        run_id += 1
                        written += 1
                        if verbose and written % 50 == 0:
                            print(f"  {written} decisoes gravadas...", flush=True)
    return written


# ----------------------------------------------------------------------------
# Partidas completas (secao 9.5)
# ----------------------------------------------------------------------------
def _points(result: str, color: str) -> float | None:
    if result == "*":
        return None
    if result == "1/2-1/2":
        return 0.5
    return 1.0 if (result == "1-0") == (color == "white") else 0.0


def play_game(white: str, black: str, budget, *, game_id, seeds, factories, decisions,
              experiment_id, commit, pgn_dir, max_plies=MAX_PLIES):
    """Joga UMA partida a partir da posicao inicial padrao, sem livro de abertura.

    Usa o motor do grupo: next_state aplica lances e tambem ClaimDraw (empate reivindicado).
    """
    import chess
    from game import get_outcome, initial_state, is_terminal, next_state, to_pgn

    state = initial_state()
    seed_w, seed_b = seeds
    agents = {chess.WHITE: factories[white](seed_w), chess.BLACK: factories[black](seed_b)}
    names = {chess.WHITE: white, chess.BLACK: black}
    run_id = next_run_id(decisions.path)
    failures, status, termination, result = 0, "completed", "", "*"
    while not is_terminal(state):
        if len(state.move_stack) >= max_plies:
            status, termination = "max_plies", "max_plies"
            break
        color = state.turn
        name = names[color]
        seed = seed_w if color == chess.WHITE else seed_b
        out = run_decision(agents[color], state, budget)
        decisions.write(build_record(
            out, experiment_id=experiment_id, run_id=run_id, run_kind="game", position_id="",
            position_set="game", context="", game_id=game_id, ply=len(state.move_stack),
            agent=name, seed=seed if name in SEEDED_AGENTS else None, repetition=0,
            budget=budget, commit=commit))
        run_id += 1
        if out.status != "ok":
            failures += 1
        if out.status in ("illegal_action", "error"):    # falha: a partida NAO vira empate
            status, termination = "failure", f"{out.status}:{name}"
            break
        state = next_state(state, out.decision.action)
    if status == "completed":
        outcome = get_outcome(state)
        result, termination = outcome.result.value, outcome.reason or ""

    pgn_path = Path(pgn_dir) / f"{game_id}.pgn"
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Event": "EDB2 - projeto xadrez", "White": white, "Black": black,
               "Round": game_id, "Termination": termination}
    pgn_path.write_text(to_pgn(state, headers) + "\n\n", encoding="utf-8")
    btype, bvalue = budget_parts(budget)
    return GameRecord(
        experiment_id=experiment_id, game_id=game_id, matchup=f"{white}_vs_{black}", white=white,
        black=black, seed_white=seed_w if white in SEEDED_AGENTS else None,
        seed_black=seed_b if black in SEEDED_AGENTS else None, budget_type=btype,
        budget_value=bvalue, result=result, status=status, termination=termination,
        plies=len(state.move_stack), failures=failures,
        points_white=_points(result, "white"), points_black=_points(result, "black"),
        pgn_path=str(pgn_path), git_commit=commit)


def run_games(*, games_table, decisions, factories, experiment_id, commit, pgn_dir,
              games_per_matchup=GAMES_PER_MATCHUP, budget_ms=GAME_BUDGET_MS, seeds=SEEDS,
              verbose=True) -> int:
    """12 partidas por confronto: 6 com cada agente de brancas, seeds 2026-2028 em rodizio."""
    budget = make_budget("time", budget_ms)
    done = {r["game_id"] for r in load_rows(games_table.path)}
    played = 0
    for a, b in MATCHUPS:
        for i in range(games_per_matchup):
            white, black = (a, b) if i % 2 == 0 else (b, a)
            seed = seeds[(i // 2) % len(seeds)]
            game_id = f"{a}_vs_{b}_{i + 1:02d}"
            if game_id in done:
                continue
            rec = play_game(white, black, budget, game_id=game_id, seeds=(seed, seed),
                            factories=factories, decisions=decisions, experiment_id=experiment_id,
                            commit=commit, pgn_dir=pgn_dir)
            games_table.write(rec)
            played += 1
            if verbose:
                print(f"{game_id}: {white} x {black} -> {rec.result} ({rec.termination}, "
                      f"{rec.plies} meios-lances)", flush=True)
    return played


# ----------------------------------------------------------------------------
# Linha de comando
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decisions", help="decisoes sobre as 120 posicoes")
    d.add_argument("--positions", default="experiments/data/positions.csv")
    d.add_argument("--set", choices=("calibration", "test", "both"), required=True)
    d.add_argument("--budget-type", choices=("time", "transitions"), required=True)
    d.add_argument("--kind", choices=("time", "memory"), default="time")
    d.add_argument("--agents", default="minimax,gpn_mcts")
    d.add_argument("--reps", type=int, default=DEFAULT_REPS)
    d.add_argument("--limit", type=int, default=None, help="usar so as N primeiras posicoes (teste rapido)")
    d.add_argument("--calibration-records", default=None,
                   help="CSV de calibracao em formato de registros (so a hibrida usa)")
    d.add_argument("--out", default="experiments/results")

    g = sub.add_parser("games", help="36 partidas completas")
    g.add_argument("--calibration-records", default=None)
    g.add_argument("--out", default="experiments/results")

    a = ap.parse_args()
    out_dir = Path(a.out)
    commit = git_commit()
    factories = make_factories(a.calibration_records)

    if a.cmd == "decisions":
        from experiments.dataset import load_positions
        positions = load_positions(a.positions, a.set)[: a.limit]
        values = TIME_BUDGETS_MS if a.budget_type == "time" else TRANSITION_BUDGETS
        exp_id = f"{a.set}_{a.budget_type}_{a.kind}"
        with decision_table(out_dir / "decisions.csv") as table:
            n = run_positions(positions, a.agents.split(","), a.budget_type, values, table=table,
                              experiment_id=exp_id, run_kind=a.kind, reps=a.reps,
                              factories=factories, commit=commit)
        print(f"{n} decisoes gravadas ({exp_id}, commit {commit[:10]})")
    else:
        with decision_table(out_dir / "decisions.csv") as dec, game_table(out_dir / "games.csv") as gt:
            n = run_games(games_table=gt, decisions=dec, factories=factories,
                          experiment_id="games_300ms", commit=commit, pgn_dir=out_dir / "pgn")
        print(f"{n} partidas jogadas")


if __name__ == "__main__":
    main()
