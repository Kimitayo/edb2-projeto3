"""GPN-MCTS: MCTS com viés de números de prova por jogador (UCT-PN).

Responsável: Gustavo (Integrante 3).
Baseado em Kowalski et al., arXiv:2506.13249 (Seções 3.1 a 3.5).
Reutiliza Clock, playout, SearchStats e DEFAULT_C de algorithms/mcts.py.
Pendências desta etapa: Score-Bounded, otimização needRecalc e mobility init.
"""
from __future__ import annotations

import math
import random

import chess

from algorithms.base import Agent, Budget, Decision
from algorithms.mcts import (DEFAULT_C, PLAYOUT_LIMIT, BudgetExhausted, Clock,
                             SearchStats, playout)
from game import Action, get_outcome, is_terminal, legal_actions, next_state

INF = math.inf
PLAYERS = (chess.WHITE, chess.BLACK)
DEFAULT_CPN = 1.0
FORMULAS = ("max", "sum", "rank")  # PNMax (padrão), PNSum e PNRank
_EPS = 1e-12


# --- Números de prova por jogador ---------------------------------------------
def _leaf_pn(state: chess.Board) -> dict[bool, float]:
    """Valores de folha. Terminal: 0 para o vencedor, INF para os demais (empate
    deixa INF para ambos). Não terminal (desconhecido): 1 para ambos."""
    if is_terminal(state):
        winner = get_outcome(state).winner
        return {p: (0.0 if winner == p else INF) for p in PLAYERS}
    return {p: 1.0 for p in PLAYERS}


def _update_pn(node: "GNode", p: bool) -> bool:
    """Recalcula pn[p] de um nó expandido. Retorna True se o valor mudou.
    Nó com o jogador p a mover: OR (mínimo). Caso contrário: AND (soma).
    Filhos ainda não criados contam como pn = 1 (valor de folha desconhecida)."""
    if not node.children:
        return False  # folha: mantém o valor de _leaf_pn
    values = [ch.pn[p] for ch in node.children.values()]
    values.extend([1.0] * len(node.untried))
    new = min(values) if node.state.turn == p else sum(values)
    changed = new != node.pn[p]
    node.pn[p] = new
    return changed


def _backup_pn(path: list["GNode"]) -> None:
    """Propaga pn dos ancestrais do último nó do caminho (para cada jogador).
    Para cedo quando um valor não muda (early stop, como no BACKUP do artigo)."""
    for p in PLAYERS:
        for node in reversed(path[:-1]):
            if not _update_pn(node, p):
                break


# --- Viés UCT-PN (artigo, seção 3.2) -------------------------------------------
def _bias(pns: list[float], formula: str) -> list[float]:
    """Viés em [0, 1] para cada irmão, a partir dos pn do jogador a mover.
    Nó com pn INF recebe 0."""
    finite = [v for v in pns if v < INF]
    if not finite:
        return [0.0] * len(pns)
    if formula == "max":  # PNMax, eq. (4)
        lo, hi = min(finite), max(finite)
        return [0.0 if v == INF else 1.0 - (v - lo) / (1.0 + hi - lo) for v in pns]
    if formula == "sum":  # PNSum, eq. (5)
        total = sum(finite)
        return [0.0 if v == INF else 1.0 - v / (1.0 + total) for v in pns]
    # PNRank, eq. (3): posto denso entre valores finitos (1 = menor pn)
    distinct = sorted(set(finite))
    rank = {v: i + 1 for i, v in enumerate(distinct)}
    max_rank = len(distinct)
    return [0.0 if v == INF else 1.0 - rank[v] / max_rank for v in pns]


# --- Árvore -------------------------------------------------------------------
class GNode:
    """Nó do GPN-MCTS. `total` segue a convenção do MCTS: acumula a utilidade do
    ponto de vista de `mover` (quem escolheu o lance até este nó)."""

    __slots__ = ("state", "parent", "action", "mover", "depth", "children",
                 "untried", "terminal", "visits", "total", "exhausted", "pn")

    def __init__(self, state: chess.Board, parent: "GNode | None" = None,
                 action: Action | None = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.mover = parent.state.turn if parent is not None else None
        self.depth = 0 if parent is None else parent.depth + 1
        self.children: dict[Action, "GNode"] = {}
        self.terminal = is_terminal(state)
        self.untried: list[Action] = [] if self.terminal else list(legal_actions(state))
        self.visits = 0
        self.total = 0.0
        self.exhausted = self.terminal
        self.pn = _leaf_pn(state)


def _select(node: GNode, rng: random.Random, c: float, cpn: float,
            formula: str) -> GNode:
    """argmax de v_i + C*sqrt(ln N_p / n_i) + Cpn*PN_i (fórmula UCT-PN, eq. 2).
    O viés usa os pn de todos os irmãos; a escolha evita subárvores exauridas."""
    all_kids = list(node.children.values())
    biases = dict(zip(map(id, all_kids),
                      _bias([ch.pn[node.state.turn] for ch in all_kids],
                            formula)))
    pool = [ch for ch in all_kids if not ch.exhausted] or all_kids
    log_n = math.log(node.visits)
    best: list[GNode] = []
    best_score = -INF
    for ch in pool:
        score = (ch.total / ch.visits
                 + c * math.sqrt(log_n / ch.visits)
                 + cpn * biases[id(ch)])
        if score > best_score + _EPS:
            best, best_score = [ch], score
        elif score >= best_score - _EPS:
            best.append(ch)
    if len(best) == 1:
        return best[0]
    return rng.choice(sorted(best, key=lambda ch: ch.action.uci()))


def _root_solved(root: GNode) -> bool:
    """Raiz provada: o jogador a mover vence (pn = 0) ou perde (pn do adversário = 0)."""
    p = root.state.turn
    return root.pn[p] == 0 or root.pn[not p] == 0


# --- Agente -------------------------------------------------------------------
class GPNMCTSAgent(Agent):
    """GPN-MCTS (UCT-PN com números de prova por jogador), sem Score-Bounded."""
    name = "gpn-mcts"

    def __init__(self, seed: int = 0, c: float = DEFAULT_C,
                 cpn: float = DEFAULT_CPN, formula: str = "max",
                 playout_limit: int = PLAYOUT_LIMIT) -> None:
        if formula not in FORMULAS:
            raise ValueError(f"Fórmula inválida {formula!r}; use {FORMULAS}")
        self.seed = seed
        self.c = c
        self.cpn = cpn
        self.formula = formula
        self.playout_limit = playout_limit

    def choose_action(self, state: chess.Board, budget: Budget) -> Decision:
        if is_terminal(state):
            raise ValueError("Não há ação a escolher em estado terminal.")

        clock = Clock(budget)
        rng = random.Random(self.seed)
        root = GNode(state)
        stats = SearchStats()

        try:
            while not root.exhausted and not clock.exhausted() and not _root_solved(root):
                self._iterate(root, rng, clock, stats)
        except BudgetExhausted:
            pass  # iteração interrompida: não é retropropagada nem anexada

        p = state.turn
        kids = [ch for ch in root.children.values() if ch.visits > 0]
        if kids:
            proven = [ch for ch in kids if ch.pn[p] == 0]  # vitória provada tem prioridade
            pool = proven or kids
            best = min(pool, key=lambda ch: (-ch.visits, ch.action.uci()))
            action, fallback = best.action, False
        else:
            action = min(legal_actions(state), key=lambda a: a.uci())
            fallback = True

        root_pn = root.pn[p]
        return Decision(
            action=action,
            nodes_expanded=stats.nodes_expanded,
            nodes_visited=stats.nodes_visited,
            max_depth_reached=stats.max_depth,
            extra={
                "algorithm": "gpn-mcts",
                "formula": self.formula,
                "cpn": self.cpn,
                "iterations": stats.iterations,
                "playouts": stats.playouts,
                "playout_steps": stats.playout_steps,
                "transitions": clock.transitions,
                "tree_nodes": 1 + stats.nodes_expanded,
                "root_visits": root.visits,
                "root_pn": None if root_pn == INF else root_pn,
                "root_solved": _root_solved(root),
                "time_s": clock.elapsed(),
                "fallback": fallback,
            },
        )

    def _iterate(self, root: GNode, rng: random.Random, clock: Clock,
                 stats: SearchStats) -> None:
        """Seleção UCT-PN, expansão, playout, retropropagação de pn e de visitas."""
        node = root
        path = [root]
        stats.nodes_visited += 1

        # 1) Seleção
        while not node.terminal and not node.untried:
            node = _select(node, rng, self.c, self.cpn, self.formula)
            path.append(node)
            stats.nodes_visited += 1

        # 2) Expansão: o filho é criado mas só é anexado após o playout,
        #    para que um corte de orçamento não deixe nós sem estatística.
        child = None
        action = None
        if node.untried:
            clock.check()
            action = node.untried.pop(rng.randrange(len(node.untried)))
            child = GNode(next_state(node.state, action), node, action)
            leaf = child
        else:
            leaf = node

        # 3) Playout (heurística H nunca vira prova)
        value, steps = playout(leaf.state, rng, clock, self.playout_limit)
        stats.playouts += 1
        stats.playout_steps += steps

        # 4) Anexa o filho e retropropaga pn
        if child is not None:
            node.children[action] = child
            path.append(child)
            stats.nodes_expanded += 1
            stats.nodes_visited += 1
            stats.max_depth = max(stats.max_depth, child.depth)
            _backup_pn(path)

        # 5) Retropropagação das visitas e valores (como no MCTS clássico)
        for n in reversed(path):
            n.visits += 1
            if n.mover is not None:
                n.total += value if n.mover == chess.WHITE else -value
            n.exhausted = n.terminal or (
                not n.untried and all(ch.exhausted for ch in n.children.values())
            )

        stats.iterations += 1
