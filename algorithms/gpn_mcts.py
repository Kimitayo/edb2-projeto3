"""GPN-MCTS: MCTS com viés de números de prova por jogador (UCT-PN).

Responsável: Gustavo (Integrante 3).
Baseado em Kowalski, Soemers, Kosakowski & Winands, "Generalized Proof-Number
Monte-Carlo Tree Search" (ECAI 2025, DOI 10.3233/FAIA251406; arXiv:2506.13249).
Variante fixa: Seção 7.2 de docs/PLANO_PROJETO_XADREZ_EDBII.md.
Reutiliza Clock, playout, SearchStats e DEFAULT_C de algorithms/mcts.py.

Configuração padrão = variante definida no planejamento do grupo:
PNRank, C = sqrt(2), Cpn = 1.0, Score-Bounded habilitado.

Componentes:
  * números de prova por jogador (pn[p]) e viés UCT-PN (PNRank/PNMax/PNSum);
  * Score-Bounded MCTS (Cazenave & Saffidine, 2010): limites pessimista/otimista
    do valor minimax de cada nó; usados para podar, resolver a raiz (inclusive
    empates, que números de prova sozinhos não provam) e alimentar os pn;
  * needRecalc: cache do viés por nó, recalculado só quando algum filho mudou
    (otimização pura: não altera a decisão, só reduz o número de ordenações do
    PNRank);
  * inicialização por mobilidade (opcional): pn inicial de folhas AND = nº de
    lances legais.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import chess

from algorithms.base import Agent, Budget, Decision
from algorithms.mcts import (DEFAULT_C, PLAYOUT_LIMIT, BudgetExhausted, Clock,
                             SearchStats, playout)
from game import (Action, get_outcome, is_terminal, legal_actions, next_state,
                  utility)

INF = math.inf
PLAYERS = (chess.WHITE, chess.BLACK)
DEFAULT_CPN = 1.0
FORMULAS = ("max", "sum", "rank")  # PNMax, PNSum e PNRank
DEFAULT_FORMULA = "rank"           # variante escolhida no planejamento (PNRank)
_EPS = 1e-12


@dataclass
class GStats(SearchStats):
    """SearchStats + métricas específicas do GPN-MCTS."""
    proof_updates: int = 0     # recálculos de pn (nó, jogador) na retropropagação
    pnrank_sorts: int = 0      # ordenações/rankings efetivamente calculados no viés
    bias_cache_hits: int = 0   # vezes em que needRecalc evitou recalcular o viés


# --- Números de prova por jogador ---------------------------------------------
def _leaf_pn(state: chess.Board, n_moves: int | None = None) -> dict[bool, float]:
    """Valores de folha. Terminal: 0 para o vencedor, INF para os demais (empate
    deixa INF para ambos). Não terminal (desconhecido): 1 para ambos.
    Com `n_moves` (inicialização por mobilidade): o jogador a mover (nó OR) fica
    com 1 e o adversário (nó AND) com o número de lances legais."""
    if is_terminal(state):
        winner = get_outcome(state).winner
        return {p: (0.0 if winner == p else INF) for p in PLAYERS}
    if n_moves is None:
        return {p: 1.0 for p in PLAYERS}
    return {p: (1.0 if state.turn == p else float(n_moves)) for p in PLAYERS}


def _update_pn(node: "GNode", p: bool, stats: GStats | None = None) -> bool:
    """Recalcula pn[p] de um nó expandido. Retorna True se o valor mudou.
    Nó com o jogador p a mover: OR (mínimo). Caso contrário: AND (soma).
    Filhos ainda não criados contam como pn = 1 (valor de folha desconhecida)."""
    if not node.children:
        return False  # folha: mantém o valor de _leaf_pn
    if stats is not None:
        stats.proof_updates += 1
    values = [ch.pn[p] for ch in node.children.values()]
    values.extend([1.0] * len(node.untried))
    new = min(values) if node.state.turn == p else sum(values)
    changed = new != node.pn[p]
    node.pn[p] = new
    return changed


# --- Score-Bounded: limites do valor minimax (ótica das brancas) ---------------
def _update_bounds(node: "GNode") -> bool:
    """Recalcula (pess, opti) de um nó expandido a partir dos filhos.
    Brancas a mover: MAX; pretas: MIN. Filhos não criados contribuem [-1, +1].
    Retorna True se algum limite mudou."""
    if not node.children:
        return False
    pess = [ch.pess for ch in node.children.values()]
    opti = [ch.opti for ch in node.children.values()]
    if node.untried:
        pess.append(-1.0)
        opti.append(1.0)
    agg = max if node.state.turn == chess.WHITE else min
    new_pess, new_opti = agg(pess), agg(opti)
    changed = new_pess != node.pess or new_opti != node.opti
    node.pess, node.opti = new_pess, new_opti
    return changed


def _sync_pn_with_bounds(node: "GNode") -> None:
    """Os limites sabem coisas que o pn sozinho não sabe. Se o jogador já tem
    vitória garantida: pn = 0. Se não consegue mais vencer (inclui empates
    provados): pn = INF."""
    proven = {chess.WHITE: node.pess >= 1.0, chess.BLACK: node.opti <= -1.0}
    possible = {chess.WHITE: node.opti >= 1.0, chess.BLACK: node.pess <= -1.0}
    for p in PLAYERS:
        if proven[p]:
            node.pn[p] = 0.0
        elif not possible[p]:
            node.pn[p] = INF


def _bounds_solved(node: "GNode") -> bool:
    """Valor minimax exato conhecido (limites coincidem)."""
    return node.pess >= node.opti


def _mover_bounds(node: "GNode", p: bool) -> tuple[float, float]:
    """(garantido, melhor caso) do ponto de vista do jogador p."""
    if p == chess.WHITE:
        return node.pess, node.opti
    return -node.opti, -node.pess


def _dominated(parent: "GNode", ch: "GNode") -> bool:
    """Poda do Score-Bounded: o filho não pode superar o que o pai já garante
    por outro lance (desigualdade estrita, nunca poda o lance que dá o limite)."""
    if parent.state.turn == chess.WHITE:
        return ch.opti < parent.pess
    return ch.pess > parent.opti


# --- Retropropagação de prova --------------------------------------------------
def _refresh(node: "GNode", score_bounded: bool,
             stats: GStats | None = None) -> bool:
    """Atualiza pn (e limites, se habilitado) de um nó. True se algo mudou."""
    before = (node.pn[chess.WHITE], node.pn[chess.BLACK], node.pess, node.opti)
    for p in PLAYERS:
        _update_pn(node, p, stats)
    if score_bounded:
        _update_bounds(node)
        _sync_pn_with_bounds(node)
    after = (node.pn[chess.WHITE], node.pn[chess.BLACK], node.pess, node.opti)
    return before != after


def _backup_pn(path: list["GNode"], stats: GStats | None = None,
               score_bounded: bool = False) -> None:
    """Propaga pn (e limites Score-Bounded) dos ancestrais do último nó do
    caminho. Para cedo quando um nó não muda (early stop, como no BACKUP do
    artigo). Marca needRecalc no pai de todo nó que mudou, pois o viés do pai
    depende do pn dos filhos."""
    for i in range(len(path) - 2, -1, -1):
        node = path[i]
        if not _refresh(node, score_bounded, stats):
            break
        if i > 0:
            path[i - 1].need_recalc = True


# --- Viés UCT-PN (artigo, seção 3.2) -------------------------------------------
def _bias(pns: list[float], formula: str,
          stats: GStats | None = None) -> list[float]:
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
    if stats is not None:
        stats.pnrank_sorts += 1
    distinct = sorted(set(finite))
    rank = {v: i + 1 for i, v in enumerate(distinct)}
    max_rank = len(distinct)
    return [0.0 if v == INF else 1.0 - rank[v] / max_rank for v in pns]


# --- Árvore -------------------------------------------------------------------
class GNode:
    """Nó do GPN-MCTS. `total` segue a convenção do MCTS: acumula a utilidade do
    ponto de vista de `mover` (quem escolheu o lance até este nó).
    `pess`/`opti`: limites do valor minimax, ótica das brancas, em [-1, 1]."""

    __slots__ = ("state", "parent", "action", "mover", "depth", "children",
                 "untried", "terminal", "visits", "total", "exhausted", "pn",
                 "pess", "opti", "need_recalc", "biases")

    def __init__(self, state: chess.Board, parent: "GNode | None" = None,
                 action: Action | None = None,
                 mobility_init: bool = False) -> None:
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
        self.pn = _leaf_pn(state, len(self.untried) if mobility_init else None)
        if self.terminal:
            self.pess = self.opti = float(utility(state, chess.WHITE))
        else:
            self.pess, self.opti = -1.0, 1.0
        self.need_recalc = True                  # needRecalc: viés ainda não calculado
        self.biases: dict[int, float] | None = None


def _child_biases(node: GNode, kids: list[GNode], formula: str,
                  stats: GStats | None, use_cache: bool) -> dict[int, float]:
    """Viés de cada filho. Com needRecalc, reaproveita o cálculo anterior enquanto
    nenhum filho tiver sido criado ou mudado de pn desde então."""
    if use_cache and not node.need_recalc and node.biases is not None:
        if stats is not None:
            stats.bias_cache_hits += 1
        return node.biases
    values = _bias([ch.pn[node.state.turn] for ch in kids], formula, stats)
    node.biases = dict(zip(map(id, kids), values))
    node.need_recalc = False
    return node.biases


def _select(node: GNode, rng: random.Random, c: float, cpn: float,
            formula: str, stats: GStats | None = None,
            score_bounded: bool = False, need_recalc: bool = True) -> GNode:
    """argmax de v_i + C*sqrt(ln N_p / n_i) + Cpn*PN_i (fórmula UCT-PN, eq. 2).
    O viés usa os pn de todos os irmãos; a escolha evita subárvores exauridas e,
    com Score-Bounded, filhos dominados por um irmão."""
    all_kids = list(node.children.values())
    biases = _child_biases(node, all_kids, formula, stats, need_recalc)
    pool = [ch for ch in all_kids if not ch.exhausted]
    if score_bounded:
        pool = [ch for ch in pool if not _dominated(node, ch)] or pool
    pool = pool or all_kids
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
    """Raiz resolvida: o jogador a mover vence (pn = 0), perde (pn do adversário
    = 0) ou, com Score-Bounded, o valor exato é conhecido (inclui empate)."""
    p = root.state.turn
    return root.pn[p] == 0 or root.pn[not p] == 0 or _bounds_solved(root)


def _decision_pool(root: GNode, kids: list[GNode], p: bool,
                   score_bounded: bool) -> list[GNode]:
    """Candidatos à jogada final. Regra do plano (Seção 7.2): filho com mais
    visitas. Única exceção, com a raiz PROVADA: vitória provada tem prioridade e,
    com Score-Bounded, só lances que garantem o valor exato da raiz (empate ou
    derrota inevitável). Raiz não resolvida: todos os filhos visitados."""
    proven = [ch for ch in kids if ch.pn[p] == 0]
    if proven:
        return proven
    if score_bounded and _bounds_solved(root):
        best_guaranteed = max(_mover_bounds(ch, p)[0] for ch in kids)
        pool = [ch for ch in kids
                if _mover_bounds(ch, p)[0] >= best_guaranteed - _EPS]
        return pool or kids
    return kids


# --- Agente -------------------------------------------------------------------
class GPNMCTSAgent(Agent):
    """GPN-MCTS (UCT-PN com números de prova por jogador).

    Padrão = planejamento do grupo: PNRank, C = sqrt(2), Cpn = 1.0 e
    Score-Bounded habilitado. `need_recalc` é só otimização (não muda o
    resultado). `mobility_init` altera a busca e fica desligado por padrão."""
    name = "gpn-mcts"

    def __init__(self, seed: int = 0, c: float = DEFAULT_C,
                 cpn: float = DEFAULT_CPN, formula: str = DEFAULT_FORMULA,
                 playout_limit: int = PLAYOUT_LIMIT,
                 score_bounded: bool = True, need_recalc: bool = True,
                 mobility_init: bool = False) -> None:
        if formula not in FORMULAS:
            raise ValueError(f"Fórmula inválida {formula!r}; use {FORMULAS}")
        self.seed = seed
        self.c = c
        self.cpn = cpn
        self.formula = formula
        self.playout_limit = playout_limit
        self.score_bounded = score_bounded
        self.need_recalc = need_recalc
        self.mobility_init = mobility_init

    def choose_action(self, state: chess.Board, budget: Budget) -> Decision:
        if is_terminal(state):
            raise ValueError("Não há ação a escolher em estado terminal.")

        clock = Clock(budget)
        rng = random.Random(self.seed)
        root = GNode(state, mobility_init=self.mobility_init)
        stats = GStats()

        try:
            while not root.exhausted and not clock.exhausted() and not _root_solved(root):
                self._iterate(root, rng, clock, stats)
        except BudgetExhausted:
            pass  # iteração interrompida: não é retropropagada nem anexada

        p = state.turn
        kids = [ch for ch in root.children.values() if ch.visits > 0]
        if kids:
            pool = _decision_pool(root, kids, p, self.score_bounded)
            best = min(pool, key=lambda ch: (-ch.visits, ch.action.uci()))
            action, fallback = best.action, False
        else:
            action = min(legal_actions(state), key=lambda a: a.uci())
            fallback = True

        root_pn = root.pn[p]
        return Decision(
            action=action,
            nodes_visited=stats.nodes_visited,
            nodes_expanded=stats.nodes_expanded,
            transitions=clock.transitions,
            max_depth_reached=stats.max_depth,
            extra={
                "algorithm": "gpn-mcts",
                "formula": self.formula,
                "c": self.c,
                "cpn": self.cpn,
                "score_bounded": self.score_bounded,
                "need_recalc": self.need_recalc,
                "mobility_init": self.mobility_init,
                "iterations": stats.iterations,
                "playouts": stats.playouts,
                "playout_steps": stats.playout_steps,
                "proof_updates": stats.proof_updates,
                "pnrank_sorts": stats.pnrank_sorts,
                "bias_cache_hits": stats.bias_cache_hits,
                "tree_nodes_created": stats.tree_nodes_created,
                "root_visits": root.visits,
                "root_pn": None if root_pn == INF else root_pn,
                "root_bounds": (root.pess, root.opti),
                "root_solved": _root_solved(root),
                **clock.common_extra(),
                "fallback": fallback,
            },
        )

    def _iterate(self, root: GNode, rng: random.Random, clock: Clock,
                 stats: GStats) -> None:
        """Seleção UCT-PN, expansão, playout, retropropagação de prova e visitas."""
        node = root
        path = [root]
        stats.nodes_visited += 1

        # 1) Seleção
        while not node.terminal and not node.untried:
            node = _select(node, rng, self.c, self.cpn, self.formula, stats,
                           self.score_bounded, self.need_recalc)
            path.append(node)
            stats.nodes_visited += 1

        # 2) Expansão: o filho é criado mas só é anexado após o playout,
        #    para que um corte de orçamento não deixe nós sem estatística.
        child = None
        action = None
        if node.untried:
            clock.check()
            action = node.untried.pop(rng.randrange(len(node.untried)))
            child = GNode(next_state(node.state, action), node, action,
                          self.mobility_init)
            leaf = child
        else:
            leaf = node

        # 3) Playout (heurística H nunca vira prova nem limite)
        inicio_playout = clock.transitions
        try:
            value, _ = playout(leaf.state, rng, clock, self.playout_limit)
        finally:
            stats.playout_steps += clock.transitions - inicio_playout

        stats.playouts += 1

        # 4) Anexa o filho e retropropaga pn (e limites Score-Bounded)
        if child is not None:
            if not node.children:
                stats.nodes_expanded += 1

            node.children[action] = child
            stats.tree_nodes_created += 1
            node.need_recalc = True
            path.append(child)
            stats.nodes_visited += 1
            stats.max_depth = max(stats.max_depth, child.depth)
            _backup_pn(path, stats, self.score_bounded)

        # 5) Retropropagação das visitas e valores (como no MCTS clássico)
        for n in reversed(path):
            n.visits += 1
            if n.mover is not None:
                n.total += value if n.mover == chess.WHITE else -value
            n.exhausted = n.terminal or (
                not n.untried and all(ch.exhausted for ch in n.children.values())
            )

        stats.iterations += 1
