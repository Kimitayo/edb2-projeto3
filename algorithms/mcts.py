"""MCTS básico com UCT sobre python-chess (Integrante 3).

ATENÇÃO: isto é MCTS/UCT *comum*, não é GPN-MCTS. Ele é a base sobre a qual a
camada de números de prova do artigo (Kowalski et al., ECAI 2025) será
acrescentada. Não chame este arquivo de GPN-MCTS.

Interface de decisão combinada com o grupo:
    choose_action(board, time_budget) -> chess.Move

Orçamento: tempo em segundos (time_budget). Existe também max_iterations,
usado SOMENTE para testes reprodutíveis; nos experimentos, deixe como None.

Desvios/simplificações em relação ao MCTS "de livro" (registrar no artigo):
  - Playout aleatório com limite de plies (playout_cap). Ao atingir o limite
    sem fim de jogo, o resultado é tratado como empate (0).
  - Recompensas em +1/0/-1 (mesma escala do artigo do GPN-MCTS), com C = sqrt(2).
  - Desempate na seleção: primeiro filho de maior valor (o artigo sorteia).
  - Empate por regra (afogamento, material insuficiente, 75 lances, 5x
    repetição) é detectado via board.outcome(); empates "reclamáveis"
    (3x repetição, 50 lances) não são considerados.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import chess

WHITE_WIN, DRAW, BLACK_WIN = 1.0, 0.0, -1.0  # +1/0/-1 (escala do artigo), ponto de vista das brancas


class Node:
    """Nó da árvore de busca.

    `wins` acumula recompensa do ponto de vista de quem FEZ o lance que levou
    a este nó (player_just_moved). Assim, o pai escolhe o filho que é melhor
    para si mesmo simplesmente maximizando wins/visits.
    """

    __slots__ = ("parent", "move", "player_just_moved", "children",
                 "untried_moves", "visits", "wins")

    def __init__(self, board: chess.Board, parent: Optional["Node"] = None,
                 move: Optional[chess.Move] = None):
        self.parent = parent
        self.move = move
        # Após push(move), o turno já passou: quem moveu é o oposto de board.turn.
        self.player_just_moved = not board.turn
        self.children: list[Node] = []
        self.untried_moves: list[chess.Move] = list(board.legal_moves)
        self.visits = 0
        self.wins = 0.0

    def uct_child(self, c: float) -> "Node":
        log_n = math.log(self.visits)
        return max(
            self.children,
            key=lambda ch: ch.wins / ch.visits + c * math.sqrt(log_n / ch.visits),
        )


@dataclass
class SearchResult:
    """Resultado da busca + métricas (úteis ao Integrante 5)."""
    move: chess.Move
    iterations: int
    nodes: int            # nós criados na árvore (inclui a raiz)
    elapsed: float        # segundos
    root_visits: dict     # {move: visitas}


def _result_white_pov(board: chess.Board) -> float:
    """Resultado de um tabuleiro terminal do ponto de vista das brancas."""
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return DRAW
    return WHITE_WIN if outcome.winner == chess.WHITE else BLACK_WIN


def _playout_with_steps(board: chess.Board, rng: random.Random, cap: int):
    """Joga lances aleatórios até o fim (ou `cap` plies). Restaura o tabuleiro.

    Devolve (resultado do ponto de vista das brancas, número de lances jogados).
    """
    pushed = 0
    result = DRAW
    while pushed < cap:
        if board.is_game_over():
            result = _result_white_pov(board)
            break
        board.push(rng.choice(list(board.legal_moves)))
        pushed += 1
    else:
        # Saiu por atingir o cap: se por acaso o último lance terminou o jogo, respeite.
        if board.is_game_over():
            result = _result_white_pov(board)
    for _ in range(pushed):
        board.pop()
    return result, pushed


def _playout(board: chess.Board, rng: random.Random, cap: int) -> float:
    return _playout_with_steps(board, rng, cap)[0]


def uct_search(board: chess.Board, time_budget: float,
               c: float = math.sqrt(2), playout_cap: int = 100,
               seed: Optional[int] = None,
               max_iterations: Optional[int] = None) -> SearchResult:
    if board.is_game_over():
        raise ValueError("Posição terminal: não há lance a escolher.")

    rng = random.Random(seed)
    work = board.copy(stack=False)  # nunca alteramos o tabuleiro do chamador
    root = Node(work)
    root.player_just_moved = not work.turn
    nodes = 1
    iterations = 0
    start = time.perf_counter()
    deadline = start + time_budget

    while True:
        # Sempre faz ao menos 1 iteração; depois respeita o orçamento.
        if iterations > 0 and time.perf_counter() >= deadline:
            break
        if max_iterations is not None and iterations >= max_iterations:
            break

        node = root
        depth = 0  # quantos push() fizemos, para desfazer depois

        # 1) SELEÇÃO: desce enquanto o nó está totalmente expandido e tem filhos.
        while not node.untried_moves and node.children:
            node = node.uct_child(c)
            work.push(node.move)
            depth += 1

        # 2) EXPANSÃO: cria um filho com um lance ainda não tentado.
        if node.untried_moves:
            move = node.untried_moves.pop(rng.randrange(len(node.untried_moves)))
            work.push(move)
            depth += 1
            child = Node(work, parent=node, move=move)
            node.children.append(child)
            node = child
            nodes += 1

        # 3) SIMULAÇÃO: playout aleatório a partir deste nó.
        if work.is_game_over():
            result = _result_white_pov(work)
        else:
            result = _playout(work, rng, playout_cap)

        # 4) RETROPROPAGAÇÃO: sobe até a raiz atualizando visitas e recompensa.
        while node is not None:
            node.visits += 1
            reward = result if node.player_just_moved == chess.WHITE else -result
            node.wins += reward
            node = node.parent

        for _ in range(depth):
            work.pop()
        iterations += 1

    # Jogada final: filho mais visitado ("robust child").
    best = max(root.children, key=lambda ch: ch.visits)
    return SearchResult(
        move=best.move,
        iterations=iterations,
        nodes=nodes,
        elapsed=time.perf_counter() - start,
        root_visits={ch.move: ch.visits for ch in root.children},
    )


def choose_action(board: chess.Board, time_budget: float, **kwargs) -> chess.Move:
    """Interface padronizada do grupo: estado + orçamento (segundos) -> lance."""
    return uct_search(board, time_budget, **kwargs).move