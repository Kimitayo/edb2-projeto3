# Guia do desenvolvedor

Guia prático para quem vai programar em cima do **motor de xadrez** do projeto: Minimax (Flavio), GPN-MCTS (Gustavo), busca híbrida (Clara) e experimentos (Victor).

- Regras, parâmetros e responsabilidades: [docs/PLANO_PROJETO_XADREZ_EDBII.md](docs/PLANO_PROJETO_XADREZ_EDBII.md)
- Referência completa do motor: [docs/game_specification.md](docs/game_specification.md)
- Dúvidas ou pedidos de mudança no motor: Matheus (Integrante 1)

---

## 1. Ambiente

Padrão do grupo: **Python 3.12** + dependências do `requirements.txt` (`python-chess==1.999`, `pytest`).

```powershell
py -3.12 -m venv .venv                       # cria o ambiente (só na 1ª vez)
.venv\Scripts\activate                       # ativa (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q                          # tudo deve passar
```

- Se o PowerShell bloquear o `activate`, rode uma vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou chame o Python da venv direto: `.venv\Scripts\python.exe -m pytest -q`.
- A pasta `.venv/` não vai para o Git; cada integrante cria a sua.
- O pacote se chama `python-chess` no `requirements.txt`, mas o import é **`import chess`** (a versão 1.999 instala o `chess` 1.11).
- Rode todos os comandos **a partir da raiz do repositório**.

---

## 2. Onde fica cada coisa

| Caminho | Conteúdo | Dono |
|---|---|---|
| `game/chess_rules.py` | motor: estado, ações, regras, fim de jogo, serialização | Matheus |
| `game/positions.py` | catálogo de 20 posições de teste das regras | Matheus |
| `game/__main__.py` | demonstração visual (`python -m game`) | Matheus |
| `game/__init__.py` | exporta a interface (`from game import ...`) | Matheus |
| `game/evaluation.py` | heurística `H(s, p)` (Seção 5 do plano), **a criar** | Flavio |
| `game/features.py` | contexto da híbrida (fase, tática), **a criar** | Clara |
| `algorithms/base.py` | contrato `Agent`, `Budget`, `Decision` + `RandomAgent` | grupo |
| `algorithms/match.py` | `play_match`: partida completa entre dois agentes | grupo |
| `algorithms/minimax.py` | Minimax + Alfa-Beta | Flavio |
| `algorithms/gpn_mcts.py` | GPN-MCTS | Gustavo |
| `algorithms/hybrid.py` | seleção por Máxima Utilidade Esperada | Clara |
| `experiments/` | métricas, CSV, runner, análise | Victor |
| `tests/test_chess_rules.py` | testes do motor | Matheus |
| `docs/` | plano, especificação do motor, protocolo experimental | — |

---

## 3. Conceitos e tipos

```python
import chess
from game import ChessState, ClaimDraw, Action, Result, Outcome
```

| Nome | O que é |
|---|---|
| `ChessState` | o **estado**: subclasse de `chess.Board` com histórico (`move_stack`) e o campo `claimed_draw`. Toda a API do `python-chess` funciona nele. |
| `Action` | `chess.Move \| ClaimDraw` |
| `chess.Move` | um lance. Campos: `from_square`, `to_square`, `promotion`. Texto: `move.uci()` → `"e2e4"`, `"e7e8q"` |
| `ClaimDraw(move=None)` | reivindicação de empate. `ClaimDraw()`: a posição atual já permite. `ClaimDraw(m)`: o lance pretendido `m` permitiria (o lance **não** é jogado). Texto: `"claim"` / `"claim:e2e4"` |
| `Result` | `WHITE_WINS` (`"1-0"`), `BLACK_WINS` (`"0-1"`), `DRAW` (`"1/2-1/2"`), `ONGOING` (`"*"`) |
| `Outcome` | `result: Result`, `reason: str \| None` e a propriedade `.winner` (`chess.WHITE`, `chess.BLACK` ou `None`) |
| Jogador | **`chess.WHITE` (`True`) ou `chess.BLACK` (`False`)**. Quem joga agora: `state.turn` |

Valores de `Outcome.reason`: `"checkmate"`, `"stalemate"`, `"insufficient_material"`, `"seventyfive_moves"`, `"fivefold_repetition"`, `"claimed_draw"` ou `None` (partida em andamento).

---

## 4. A interface (tudo via `from game import ...`)

| Função | Retorno | Para que serve |
|---|---|---|
| `initial_state(fen=None)` | `ChessState` | posição inicial, ou a posição de uma FEN |
| `from_moves(moves, start_fen=None)` | `ChessState` | posição a partir de uma lista de lances UCI, **com histórico** |
| `legal_actions(state)` | `tuple[Action, ...]` | lances legais em ordem UCI, depois as reivindicações; `()` se terminal |
| `next_state(state, action)` | `ChessState` | novo estado; **o original não muda**; conta 1 transição |
| `is_terminal(state)` | `bool` | a partida acabou? |
| `get_outcome(state)` | `Outcome` | quem venceu e por quê |
| `utility(state, player)` | `float` | `+1` / `0` / `−1` para `player`; só em estado terminal |
| `reset_transition_count()` / `transition_count()` | — / `int` | contador global de transições |
| `render(state, unicode=True)` | `str` | tabuleiro em texto (depuração) |
| `render_svg(state, size=400)` | `str` | imagem SVG do tabuleiro |
| `serialize(state)` / `deserialize(data)` | `dict` / `ChessState` | salvar e recarregar estados (compatível com JSON) |
| `to_pgn(state, headers=None)` | `str` | partida em PGN |

### Exemplo completo

```python
import chess
from game import initial_state, legal_actions, next_state, is_terminal, utility, get_outcome

s = initial_state()
print(len(legal_actions(s)))                 # 20
s = next_state(s, chess.Move.from_uci("e2e4"))
while not is_terminal(s):
    s = next_state(s, legal_actions(s)[0])   # sempre a 1ª ação, só como exemplo
print(get_outcome(s))                        # Outcome(result=..., reason=...)
print(utility(s, chess.WHITE))               # +1.0, 0.0 ou -1.0
```

---

## 5. Como o motor funciona por dentro

**Imutabilidade.** `next_state` faz `state.copy(stack=True)` e aplica a ação na cópia. Você pode guardar estados em nós da árvore sem medo de alguém alterá-los. **Nunca** chame `state.push()` ou `state.pop()` num estado recebido do motor ou de outro algoritmo.

**Validação.** `next_state` lança `ValueError` para lance ilegal ou partida encerrada e `TypeError` se a ação não for `chess.Move`/`ClaimDraw`. Use sempre ações vindas de `legal_actions`.

**Contador de transições.** Cada `next_state` bem-sucedido soma 1 a um contador global (`ClaimDraw` também conta). `from_moves` e `deserialize` **não** contam. É a métrica comum do orçamento `max_transitions`. Ninguém precisa contar transições por conta própria.

**Histórico.** Repetições dependem dos lances anteriores. Por isso guardamos sempre **FEN inicial + lances** (`from_moves`, `serialize`), nunca só a FEN final.

**Fim de jogo.**

| Situação | Como termina | `reason` |
|---|---|---|
| Xeque-mate | automático | `checkmate` |
| Afogamento | automático | `stalemate` |
| Material insuficiente | automático | `insufficient_material` |
| 75 lances sem captura nem movimento de peão | automático | `seventyfive_moves` |
| 5ª repetição | automático | `fivefold_repetition` |
| 3ª repetição ou 50 lances | **só se alguém escolher `ClaimDraw`** | `claimed_draw` |

O mate tem precedência: um mate no 150º meio-lance vale como mate.

**Reivindicação de empate na busca.** `ClaimDraw` é uma ação como outra qualquer. Ela aparece em `legal_actions` **depois** dos lances e leva a um estado terminal com `utility = 0`. Se a posição atual já permite reivindicar, só `ClaimDraw()` aparece (sem as versões com lance pretendido). A detecção foi conferida contra o `python-chess` em cerca de 99 mil posições.

**Custo.** Cada `next_state` copia o histórico, então o custo cresce com o tamanho da partida. `legal_actions` só testa reivindicações lance a lance quando o contador de meios-lances é ≥ 7; antes disso, nenhuma repetição tripla é possível. Se o motor ficar lento para a sua busca, fale com o Matheus antes de contornar a interface: o plano exige que todos usem a mesma interface para a contagem ser justa.

---

## 6. Receitas por integrante

### Flavio: Minimax e heurística

```python
from game import legal_actions, next_state, is_terminal, utility

def minimax(state, depth, alpha, beta, root_player):
    if is_terminal(state):
        return utility(state, root_player)         # resultado exato sempre vence a heurística
    if depth == 0:
        return H(state, root_player)               # game/evaluation.py (Seção 5 do plano)
    maximizing = state.turn == root_player
    ...
    for action in ordered(legal_actions(state)):   # ordenação: capturas, promoções, xeques, resto, UCI
        child = next_state(state, action)
        ...
```

API útil do `python-chess` para a ordenação e a heurística:
- ordenação: `state.is_capture(m)`, `m.promotion is not None`, `state.gives_check(m)` (só para `chess.Move`; trate `ClaimDraw` à parte);
- heurística: `state.pieces(chess.PAWN, chess.WHITE)` (conjunto de casas), `state.attacks(sq)`, `state.king(cor)`, `state.is_check()`, `state.piece_map()`, `chess.square_file(sq)` / `chess.square_rank(sq)`.

### Gustavo: GPN-MCTS

```python
import random
from game import legal_actions, next_state, is_terminal, utility, get_outcome

rng = random.Random(2026)                           # seeds 2026, 2027, 2028
def playout(state, root_player, limit=80):
    for _ in range(limit):
        if is_terminal(state):
            return utility(state, root_player)      # só isto pode virar prova
        state = next_state(state, rng.choice(legal_actions(state)))
    return H(state, root_player)                    # estimativa, NUNCA prova
```

- Estado terminal: `get_outcome(state).winner` diz quem venceu (`None` = empate) e serve para os números de prova por jogador.
- Nó OR/AND: compare `state.turn` com o jogador da prova.
- `legal_actions` é determinístico (ordem UCI), então a mesma seed gera a mesma árvore.

### Clara: contexto da híbrida

Tudo sai do próprio estado (`chess.Board`):
- abertura: `state.fullmove_number <= 10`;
- final: contar peças que não são rei nem peão, por exemplo `sum(len(state.pieces(t, c)) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for c in (chess.WHITE, chess.BLACK)) <= 6`;
- tática: `state.is_check() or any(state.is_capture(m) for m in state.legal_moves)`;
- ramificação: `len(legal_actions(state))`.

A híbrida escolhe **um** algoritmo e repassa o mesmo `state`; o motor garante que ele chega intacto.

### Victor: experimentos

```python
from game import from_moves, serialize, deserialize, to_pgn, reset_transition_count, transition_count

pos = from_moves(["e2e4", "e7e5", "g1f3"])          # posição do Lichess com histórico
registro = serialize(pos)                            # vai para JSON/CSV; deserialize() recarrega

reset_transition_count()
decisao = agente.choose_action(pos, budget)
transicoes = transition_count()                      # custo da decisão
```

- Para ler PGN do Lichess: `chess.pgn.read_game(arquivo)` → `[m.uci() for m in jogo.mainline_moves()]` → `from_moves(...)`.
- Para salvar as 36 partidas: `to_pgn(estado_final, {"White": "minimax", "Black": "gpn_mcts"})`.
- `algorithms/match.py::play_match` já joga uma partida completa e registra transições e tempo de cada decisão.

---

## 7. Regras de ouro

1. Use **só** a interface de `game` para gerar sucessores (`next_state`); é ela que conta as transições.
2. Não modifique estados recebidos (`push`/`pop`/atribuições).
3. Jogador é `chess.WHITE`/`chess.BLACK`, nunca 0/1. `utility(s, 0)` lança erro de propósito.
4. `utility` só em estado terminal; em estado não terminal use a heurística `H`.
5. Heurística nunca prova vitória ou derrota; só estados terminais provam.
6. Trate `ClaimDraw` como ação válida (empate, utilidade 0) e não suponha que toda ação é `chess.Move`. Use `isinstance(a, chess.Move)` quando precisar.
7. Sempre que mudar algo, rode `python -m pytest -q` antes do commit.

---

## 8. Ferramentas de depuração

```powershell
python -m game                          # partida aleatória animada no terminal
python -m game --delay 0 --seed 7       # só o tabuleiro final
python -m game --svg final.svg --pgn partida.pgn
python -m game --ascii                  # letras em vez de símbolos Unicode
python -m game.positions                # mostra as 20 posições de teste
python -m algorithms.match              # partida RandomAgent x RandomAgent via play_match
```

No código: `print(render(state))` em qualquer ponto da busca, ou `positions.load("mate_em_1")` para pegar uma posição pronta (`from game import positions`).

---

## 9. Testes e Git

- Testes do motor: `python -m pytest tests/test_chess_rules.py -q`. Cada algoritmo deve ter o seu arquivo em `tests/` (ver Seção 12.3 do plano).
- Fluxo (Seção 12.4 do plano): branch própria a partir da `main` atualizada → commits → Pull Request → revisão de outro integrante.
- Cada experimento registra o hash do commit usado (`git rev-parse HEAD`).

---

## 10. Pendências conhecidas fora do motor

- `algorithms/base.py`: `Budget` e `Decision` ainda não seguem o contrato final da Seção 12.2 do plano (`max_time_ms`, campo `transitions`). Precisa de decisão do grupo.
- `experiments/metrics.py`, `results_storage.py` e `test_metrics.py` (Victor) ainda usam nomes da versão anterior (`BattleAggregator`, `hp_remaining_a/b`, `scenario_*`).
- `game/evaluation.py` (Flavio) e `game/features.py` (Clara) ainda não existem.
