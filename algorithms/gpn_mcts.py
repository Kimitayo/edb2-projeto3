"""GPN-MCTS (Generalized Proof-Number MCTS) com viés PNRank, sobre python-chess.

Referência: Kowalski, Soemers, Kosakowski, Winands. "Generalized Proof-Number
Monte-Carlo Tree Search", ECAI 2025 (arXiv:2506.13249). Seções 2.3, 3.1, 3.2 e
Algoritmo 1 (seção 3.5).

Mecanismos do artigo PRESERVADOS aqui:
  1. Proof numbers POR JOGADOR em cada nó (sem disproof numbers). Para o
     jogador p, um nó em que p move é OR (pn = mínimo dos filhos) e um nó em
     que o outro move é AND (pn = soma dos filhos). Folha desconhecida: pn=1;
     vitória de p: pn=0; derrota OU empate de p: pn=infinito.
  2. Seleção UCT-PN (fórmula 2):  v_i + C*sqrt(ln n_pai / n_i) + Cpn*PNRank(i)
     onde o viés usa os proof numbers do jogador que move no nó pai.
  3. PNRank (fórmula 3): 1 - rank(i)/max_rank, empates com o mesmo rank.
  4. Retropropagação dos proof numbers com parada antecipada quando nada muda
     e flag needRecalc (cálculo do viés sob demanda, seção 3.2.5).
  5. Recompensas +1/0/-1 e C = sqrt(2), como nos experimentos do artigo.
  6. Lance final: filho da raiz mais visitado.

SIMPLIFICAÇÕES / INTERPRETAÇÕES (devem ser registradas no artigo):
  - NÃO há Score Bounded MCTS (seção 3.4). O artigo avalia versões com e sem
    ele; esta é a versão sem.
  - NÃO há mobility initialization (seção 3.3) nem reaproveitamento de árvore.
  - Só PNRank; PNMax e PNSum (seções 3.2.2 e 3.2.3) não foram implementados.
  - O artigo não define o desempate de ranks. Usamos ranking denso
    (valores iguais -> mesmo rank; próximo valor -> rank seguinte).
  - Alg. 1 percorre apenas os filhos já criados ao calcular min/soma; foi
    seguido ao pé da letra (nós parcialmente expandidos usam só os filhos
    existentes).
  - A flag needRecalc é ligada no PAI do nó cujo proof number mudou (é o pai
    que precisa recalcular o viés dos filhos); o recálculo é feito só quando
    o nó vai de fato escolher entre filhos. A ordem exata no Alg. 1 estava
    ambígua na versão lida.
  - Desempate na seleção: primeiro filho (determinístico), não aleatório.
  - Playout aleatório com limite de plies; ao atingir o limite sem fim de
    jogo, o resultado é empate (mesma regra do UCT em mcts.py).
  - Parâmetros padrão (cpn=1.0, playout de 80 meios-lances) vêm do documento do
    grupo (seção 7), NÃO do artigo: o artigo mostra que o melhor Cpn varia muito
    por jogo, então o grupo não deve afirmar que 1.0 é ótimo para xadrez.
  - Escolha final: mais visitas; empate decidido pelo UCI (documento do grupo).
  - Score-Bounded MCTS OMITIDO por decisão do integrante 3: o documento do grupo
    o prevê habilitado, mas o artigo não detalha como os limites alimentam os
    proof numbers (seção 3.4 só o descreve como técnica auxiliar). Declarar como
    simplificação/diferença em relação à configuração do grupo.
  - PENDENTE: heurística H no corte do playout (definição e escala vêm do
    grupo). Hoje o corte vale empate (0).
  - Dois jogadores apenas (xadrez).
"""

from __future__ import annotations

import math
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional, Sequence

import chess

from .mcts import SearchResult, _playout_with_steps, _result_white_pov

INF = math.inf


class GPNNode:
    """Nó da árvore GPN-MCTS.

    value_sum acumula a recompensa (+1/0/-1) do ponto de vista de quem FEZ o
    lance que levou ao nó (player_just_moved), como no UCT de mcts.py.
    pn[int(p)] é o proof number do jogador p (índice 1 = brancas, 0 = pretas).
    """

    __slots__ = ("parent", "move", "player_to_move", "player_just_moved",
                 "terminal", "winner", "children", "untried", "visits",
                 "value_sum", "pn", "bias", "need_recalc")

    def __init__(self, board: chess.Board, parent: Optional["GPNNode"] = None,
                 move: Optional[chess.Move] = None):
        self.parent = parent
        self.move = move
        self.player_to_move = board.turn
        self.player_just_moved = not board.turn
        self.terminal = board.is_game_over()
        self.winner: Optional[bool] = None  # None = sem vencedor (ou não terminal)
        if self.terminal:
            outcome = board.outcome()
            self.winner = None if outcome is None else outcome.winner
        self.untried: list[chess.Move] = [] if self.terminal else list(board.legal_moves)
        self.children: list[GPNNode] = []
        self.visits = 0
        self.value_sum = 0.0
        self.pn = [1.0, 1.0]
        self.bias = 0.0            # viés PNRank deste nó, visto do pai
        self.need_recalc = False


def update_proof_number(node: GPNNode, p: bool) -> bool:
    """UPDATEPROOFNUMBER do Alg. 1. Devolve True se o valor mudou."""
    i = int(p)
    if not node.children:  # folha da árvore
        if not node.terminal:
            node.pn[i] = 1.0
        elif node.winner == p:
            node.pn[i] = 0.0
        else:  # derrota ou empate de p
            node.pn[i] = INF
        return True
    old = node.pn[i]
    if p == node.player_to_move:  # nó OR para p
        new = min(ch.pn[i] for ch in node.children)
    else:                         # nó AND para p
        new = sum(ch.pn[i] for ch in node.children)
    node.pn[i] = new
    return new != old


def pn_rank_bias(pns: Sequence[float]) -> list[float]:
    """PNRank (fórmula 3) para um conjunto de irmãos: 1 - rank/max_rank.

    Menor proof number -> rank 1. Valores iguais -> mesmo rank (ranking denso).
    Infinito fica com o maior rank e recebe viés 0.
    """
    distinct = sorted(set(pns))
    rank_of = {v: k + 1 for k, v in enumerate(distinct)}
    max_rank = len(distinct)
    return [1.0 - rank_of[v] / max_rank for v in pns]


def _update_children_bias(node: GPNNode) -> None:
    """UPDATECHILDRENPNSCORES: viés dos filhos para o jogador que move em `node`."""
    i = int(node.player_to_move)
    biases = pn_rank_bias([ch.pn[i] for ch in node.children])
    for ch, b in zip(node.children, biases):
        ch.bias = b
    node.need_recalc = False


def _best_uct_pn_child(node: GPNNode, c: float, cpn: float) -> GPNNode:
    """BESTUCTPNCHILD: maximiza a fórmula UCT-PN (fórmula 2)."""
    log_n = math.log(node.visits)
    return max(
        node.children,
        key=lambda ch: (ch.value_sum / ch.visits
                        + c * math.sqrt(log_n / ch.visits)
                        + cpn * ch.bias),
    )


@dataclass
class GPNSearchResult(SearchResult):
    root_pn: dict = field(default_factory=dict)  # {chess.WHITE: pn, chess.BLACK: pn}
    # Métricas próprias (seção 7.5 do documento do grupo). Definições:
    playouts: int = 0            # playouts executados (nó terminal não faz playout)
    playout_steps: int = 0       # lances jogados dentro dos playouts
    nodes_visited: int = 0       # nós percorridos na seleção, somados sobre as iterações
    nodes_expanded: int = 0      # nós criados por expansão (= nodes - 1)
    transitions: int = 0         # aplicações de lance (seleção + expansão + playout)
    max_depth: int = 0           # maior profundidade (em plies) de uma folha atingida
    pn_updates: int = 0          # chamadas a update_proof_number
    pn_rank_sorts: int = 0       # recálculos do PNRank (cada um ordena os filhos)
    cpu_time: float = 0.0        # segundos de CPU do processo durante a busca
    memory_peak_bytes: Optional[int] = None  # pico via tracemalloc (só se track_memory)


def gpn_mcts_search(board: chess.Board, time_budget: float, cpn: float = 1.0,
                    c: float = math.sqrt(2), playout_cap: int = 80,
                    seed: Optional[int] = None,
                    max_iterations: Optional[int] = None,
                    track_memory: bool = False) -> GPNSearchResult:
    if board.is_game_over():
        raise ValueError("Posição terminal: não há lance a escolher.")

    if track_memory:
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()

    rng = random.Random(seed)
    work = board.copy(stack=False)  # nunca alteramos o tabuleiro do chamador
    root = GPNNode(work)
    nodes = 1
    iterations = playouts = playout_steps = nodes_visited = transitions = 0
    max_depth = pn_updates = pn_rank_sorts = 0
    start = time.perf_counter()
    cpu_start = time.process_time()
    deadline = start + time_budget

    while True:
        if iterations > 0 and time.perf_counter() >= deadline:
            break
        if max_iterations is not None and iterations >= max_iterations:
            break

        node = root
        depth = 0
        nodes_visited += 1

        # TREE_POLICY: seleção (UCT-PN) + expansão.
        while not node.terminal:
            if node.untried:
                move = node.untried.pop(rng.randrange(len(node.untried)))
                work.push(move)
                depth += 1
                transitions += 1
                child = GPNNode(work, parent=node, move=move)
                node.children.append(child)
                for p in (chess.WHITE, chess.BLACK):
                    update_proof_number(child, p)
                    pn_updates += 1
                node = child
                nodes += 1
                nodes_visited += 1
                break
            if node.need_recalc:
                _update_children_bias(node)
                pn_rank_sorts += 1
            node = _best_uct_pn_child(node, c, cpn)
            work.push(node.move)
            depth += 1
            transitions += 1
            nodes_visited += 1
        max_depth = max(max_depth, depth)

        # DEFAULT_POLICY: resultado em +1 / 0 / -1 do ponto de vista das brancas
        # (mcts.py já usa essa escala).
        if node.terminal:
            result = _result_white_pov(work)
        else:
            result, steps = _playout_with_steps(work, rng, playout_cap)
            playouts += 1
            playout_steps += steps
            transitions += steps

        # BACKUP (a): visitas e recompensa até a raiz.
        v = node
        while v is not None:
            v.visits += 1
            v.value_sum += result if v.player_just_moved == chess.WHITE else -result
            v = v.parent

        # BACKUP (b): proof numbers, um jogador por vez, com parada antecipada.
        for p in (chess.WHITE, chess.BLACK):
            v = node
            changed = True
            while changed and v is not None:
                changed = update_proof_number(v, p)
                pn_updates += 1
                if changed and v.parent is not None:
                    v.parent.need_recalc = True
                v = v.parent

        for _ in range(depth):
            work.pop()
        iterations += 1

    # Escolha final: mais visitas; empate decidido pelo UCI (ordem lexicográfica).
    best = min(root.children, key=lambda ch: (-ch.visits, ch.move.uci()))

    memory_peak = None
    if track_memory:
        memory_peak = tracemalloc.get_traced_memory()[1]
        if not was_tracing:
            tracemalloc.stop()

    return GPNSearchResult(
        move=best.move,
        iterations=iterations,
        nodes=nodes,
        elapsed=time.perf_counter() - start,
        root_visits={ch.move: ch.visits for ch in root.children},
        root_pn={chess.WHITE: root.pn[1], chess.BLACK: root.pn[0]},
        playouts=playouts,
        playout_steps=playout_steps,
        nodes_visited=nodes_visited,
        nodes_expanded=nodes - 1,
        transitions=transitions,
        max_depth=max_depth,
        pn_updates=pn_updates,
        pn_rank_sorts=pn_rank_sorts,
        cpu_time=time.process_time() - cpu_start,
        memory_peak_bytes=memory_peak,
    )


def choose_action(board: chess.Board, time_budget: float, **kwargs) -> chess.Move:
    """Interface padronizada do grupo: estado + orçamento (segundos) -> lance."""
    return gpn_mcts_search(board, time_budget, **kwargs).move