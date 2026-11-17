import math
import os
import sys

import chess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algorithms.gpn_mcts import (  # noqa: E402
    INF, GPNNode, choose_action, gpn_mcts_search, pn_rank_bias,
    update_proof_number,
)
from algorithms.mcts import uct_search  # noqa: E402

MATE_BRANCAS = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"   # Ra8#
MATE_PRETAS = "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"    # ...Ra1#


# ---------- PNRank (fórmula 3 e propriedades (a), (b), (c) da seção 3.2.1) ----------

def test_pnrank_valores_conhecidos():
    # ranks densos: 1 -> 1, 3 -> 2 (empate), inf -> 3
    assert pn_rank_bias([1, 3, 3, INF]) == pytest.approx([2 / 3, 1 / 3, 1 / 3, 0.0])


def test_pnrank_propriedade_a_infinito_vale_zero():
    assert pn_rank_bias([2, INF, 5])[1] == 0.0


def test_pnrank_propriedade_b_finitos_positivos_se_ha_infinito():
    for pns in ([1, INF], [4, 4, INF], [1, 2, 3, INF, INF]):
        out = pn_rank_bias(pns)
        for v, b in zip(pns, out):
            if v != INF:
                assert b > 0


def test_pnrank_propriedade_c_iguais_recebem_o_mesmo_valor():
    out = pn_rank_bias([2, 7, 2, 7, INF])
    assert out[0] == out[2] and out[1] == out[3]


def test_pnrank_todos_iguais_nao_enviesa():
    assert pn_rank_bias([1, 1, 1]) == [0.0, 0.0, 0.0]


def test_pnrank_menor_pn_tem_maior_vies():
    out = pn_rank_bias([5, 1, 3])
    assert out[1] > out[2] > out[0]


# ---------- regras OR/AND dos proof numbers por jogador ----------

def _arvore_um_nivel(fen):
    """Raiz com todos os filhos criados à mão e proof numbers de folha inicializados."""
    board = chess.Board(fen)
    root = GPNNode(board)
    for m in list(board.legal_moves):
        board.push(m)
        ch = GPNNode(board, parent=root, move=m)
        root.children.append(ch)
        board.pop()
        for p in (chess.WHITE, chess.BLACK):
            update_proof_number(ch, p)
    return root


def test_folha_terminal_com_vitoria_de_p():
    board = chess.Board(MATE_BRANCAS)
    board.push(chess.Move.from_uci("a1a8"))
    n = GPNNode(board)
    update_proof_number(n, chess.WHITE)
    update_proof_number(n, chess.BLACK)
    assert n.pn[int(chess.WHITE)] == 0.0      # vitória de p: 0
    assert n.pn[int(chess.BLACK)] == INF      # derrota de p: infinito


def test_folha_empate_vale_infinito_para_os_dois():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # afogamento
    n = GPNNode(board)
    update_proof_number(n, chess.WHITE)
    update_proof_number(n, chess.BLACK)
    assert n.pn == [INF, INF]


def test_folha_desconhecida_vale_um():
    n = GPNNode(chess.Board())
    update_proof_number(n, chess.WHITE)
    update_proof_number(n, chess.BLACK)
    assert n.pn == [1.0, 1.0]


def test_no_or_usa_minimo_e_no_and_usa_soma():
    root = _arvore_um_nivel(MATE_BRANCAS)  # brancas movem na raiz
    k = len(root.children)
    # Para as brancas a raiz é OR: existe um filho com pn=0 (Ra8#) -> mínimo 0.
    update_proof_number(root, chess.WHITE)
    assert root.pn[int(chess.WHITE)] == 0.0
    # Para as pretas a raiz é AND: soma dos pn das pretas nos filhos;
    # o filho com mate tem pn[pretas]=inf, logo a soma é inf.
    update_proof_number(root, chess.BLACK)
    assert root.pn[int(chess.BLACK)] == INF
    # Sem o filho de mate, a soma das pretas seria k-1 (cada folha desconhecida vale 1).
    soma_sem_mate = sum(ch.pn[int(chess.BLACK)] for ch in root.children
                        if ch.pn[int(chess.BLACK)] != INF)
    assert soma_sem_mate == k - 1


def test_atualizacao_devolve_se_houve_mudanca():
    root = _arvore_um_nivel(MATE_BRANCAS)
    assert update_proof_number(root, chess.WHITE) is True    # 1 -> 0
    assert update_proof_number(root, chess.WHITE) is False   # já estava em 0


# ---------- comportamento da busca ----------

def test_retorna_lance_legal_e_nao_altera_o_tabuleiro():
    board = chess.Board()
    antes = board.fen()
    move = choose_action(board, time_budget=0.3, cpn=1.0, seed=1)
    assert move in board.legal_moves
    assert board.fen() == antes


def test_encontra_mate_em_1_brancas_e_pretas():
    assert choose_action(chess.Board(MATE_BRANCAS), 1.0, cpn=1.0, seed=42) == chess.Move.from_uci("a1a8")
    assert choose_action(chess.Board(MATE_PRETAS), 1.0, cpn=1.0, seed=42) == chess.Move.from_uci("a8a1")


def test_raiz_prova_vitoria_no_mate_em_1():
    r = gpn_mcts_search(chess.Board(MATE_BRANCAS), 60, cpn=1.0, seed=5, max_iterations=600)
    assert r.root_pn[chess.WHITE] == 0.0
    r = gpn_mcts_search(chess.Board(MATE_PRETAS), 60, cpn=1.0, seed=5, max_iterations=600)
    assert r.root_pn[chess.BLACK] == 0.0


def test_posicao_terminal_levanta_erro():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    with pytest.raises(ValueError):
        choose_action(board, time_budget=0.1, cpn=1.0)


def test_reprodutivel_com_semente_e_iteracoes_fixas():
    board = chess.Board()
    r1 = gpn_mcts_search(board, 60, cpn=1.0, seed=7, max_iterations=300)
    r2 = gpn_mcts_search(board, 60, cpn=1.0, seed=7, max_iterations=300)
    assert r1.root_visits == r2.root_visits


def test_respeita_o_orcamento_de_tempo():
    r = gpn_mcts_search(chess.Board(), 0.5, cpn=1.0, seed=1)
    assert 0.5 <= r.elapsed < 0.8
    assert r.iterations > 0 and r.nodes > 1


def test_visitas_na_raiz_somam_as_iteracoes():
    r = gpn_mcts_search(chess.Board(), 60, cpn=1.0, seed=3, max_iterations=200)
    assert sum(r.root_visits.values()) == 200


# ---------- regressão: com Cpn = 0 o GPN-MCTS deve se comportar como o UCT ----------

@pytest.mark.parametrize("c", [math.sqrt(2), 0.7])
@pytest.mark.parametrize("fen", [chess.STARTING_FEN, MATE_BRANCAS,
                                 "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"])
def test_cpn_zero_equivale_ao_uct(fen, c):
    # Ambos usam recompensas +1/0/-1 (escala do artigo) e o mesmo C: com Cpn = 0
    # o viés some e as buscas devem ser idênticas (mesma semente, mesmas iterações).
    board = chess.Board(fen)
    g = gpn_mcts_search(board, 60, cpn=0.0, c=c, playout_cap=100, seed=11, max_iterations=400)
    u = uct_search(board, 60, c=c, playout_cap=100, seed=11, max_iterations=400)
    assert g.root_visits == u.root_visits


def test_parametros_padrao_do_documento_do_grupo():
    import inspect
    params = inspect.signature(gpn_mcts_search).parameters
    assert params["cpn"].default == 1.0
    assert params["playout_cap"].default == 80
    assert params["c"].default == pytest.approx(math.sqrt(2))


# ---------- escolha final: mais visitas, empate por UCI ----------

def test_desempate_por_uci():
    # Com max_iterations = nº de lances legais, a raiz expande um filho por
    # iteração antes de qualquer seleção: todos ficam com 1 visita (empate).
    board = chess.Board()
    n = board.legal_moves.count()
    r = gpn_mcts_search(board, 60, seed=5, max_iterations=n)
    assert set(r.root_visits.values()) == {1}
    assert r.move.uci() == min(m.uci() for m in board.legal_moves)


# ---------- métricas (seção 7.5) ----------

def test_metricas_sao_consistentes():
    r = gpn_mcts_search(chess.Board(), 60, seed=3, max_iterations=300)
    assert r.iterations == 300
    assert r.nodes_expanded == r.nodes - 1
    assert 0 < r.playouts <= r.iterations
    assert r.playout_steps <= r.playouts * 80          # limite de 80 meios-lances
    assert r.transitions >= r.playout_steps
    assert r.nodes_visited >= r.iterations
    assert r.max_depth >= 1
    assert r.pn_updates > 0 and r.pn_rank_sorts > 0
    assert r.cpu_time > 0
    assert r.memory_peak_bytes is None                 # só mede se pedirmos


def test_memoria_so_e_medida_quando_pedida():
    r = gpn_mcts_search(chess.Board(), 60, seed=3, max_iterations=100, track_memory=True)
    assert r.memory_peak_bytes is not None and r.memory_peak_bytes > 0


def test_playout_respeita_o_limite_de_meios_lances():
    r = gpn_mcts_search(chess.Board(), 60, seed=2, max_iterations=150, playout_cap=10)
    assert r.playout_steps <= r.playouts * 10