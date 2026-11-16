# Especificação do motor de xadrez e interface comum

**Responsável:** Integrante 1 (Matheus Bulhões)
**Código:** [`game/chess_rules.py`](../game/chess_rules.py) · **Testes:** [`tests/test_chess_rules.py`](../tests/test_chess_rules.py)
**Base:** [PLANO_PROJETO_XADREZ_EDBII.md](PLANO_PROJETO_XADREZ_EDBII.md), Seções 3 e 4.

Xadrez padrão com a biblioteca `python-chess` (`python-chess==1.999`, que instala o pacote `chess` 1.11). No código, o import é `import chess`.

## 1. Uso rápido

```python
import chess
from game import initial_state, legal_actions, next_state, is_terminal, get_outcome, utility

s = initial_state()                       # posição inicial
while not is_terminal(s):
    action = legal_actions(s)[0]          # chess.Move ou ClaimDraw
    s = next_state(s, action)             # novo estado; o anterior não muda
print(get_outcome(s), utility(s, chess.WHITE))
```

## 2. Convenções

| Conceito | Representação |
|---|---|
| Estado | `ChessState`, subclasse de `chess.Board` com histórico (`move_stack`) e o campo `claimed_draw` |
| Jogador | `chess.WHITE` (`True`) ou `chess.BLACK` (`False`). **Não** use 0/1: `utility` rejeita esses valores |
| Ação | `Action = chess.Move \| ClaimDraw` |
| Transição | uma chamada bem-sucedida a `next_state` (inclusive `ClaimDraw`) |
| Utilidade | `+1` vitória, `0` empate, `−1` derrota, só em estados terminais |

O estado recebido **nunca** é modificado: `next_state` copia com `copy(stack=True)` e aplica a ação na cópia. Como `ChessState` é um `chess.Board`, toda a API da biblioteca continua disponível (`is_check()`, `piece_map()`, `attacks()`...). `next_state` também aceita um `chess.Board` comum e sempre devolve um `ChessState`.

## 3. Interface

| Função | Retorno | Observações |
|---|---|---|
| `initial_state(fen=None)` | `ChessState` | `ValueError` se a FEN for inválida ou ilegal |
| `from_moves(moves, start_fen=None)` | `ChessState` | lances UCI ou `chess.Move`; preserva o histórico; não conta transições |
| `legal_actions(state)` | `tuple[Action, ...]` | lances em ordem UCI, seguidos das reivindicações; `()` se terminal |
| `next_state(state, action)` | `ChessState` | `ValueError` se ilegal ou partida encerrada; `TypeError` se não for ação |
| `is_terminal(state)` | `bool` | ver Seção 5 |
| `get_outcome(state)` | `Outcome` | `result` (`Result`), `reason` e `.winner` |
| `utility(state, player)` | `float` | `ValueError` se não terminal ou jogador inválido |
| `render(state, unicode=True)` | `str` | tabuleiro com coordenadas e linha de informações |
| `render_svg(state, size=400)` | `str` | imagem com seta no último lance e xeque destacado |
| `serialize(state)` / `deserialize(data)` | `dict` / `ChessState` | compatível com JSON; reconstrói reaplicando os lances |
| `to_pgn(state, headers=None)` | `str` | PGN com o resultado do motor |
| `reset_transition_count()` / `transition_count()` | — / `int` | contador global de transições |

`Result`: `WHITE_WINS` (`"1-0"`), `BLACK_WINS` (`"0-1"`), `DRAW` (`"1/2-1/2"`), `ONGOING` (`"*"`).
`Outcome.reason`: `checkmate`, `stalemate`, `insufficient_material`, `seventyfive_moves`, `fivefold_repetition`, `claimed_draw` ou `None` (em andamento).

## 4. Reivindicação de empate (`ClaimDraw`)

Os empates **reivindicáveis** (3ª repetição e regra dos 50 lances) não encerram a partida sozinhos: o jogador da vez precisa escolher a ação `ClaimDraw`.

- `ClaimDraw()`: a posição atual já permite reivindicar.
- `ClaimDraw(move)`: o lance pretendido `move` tornaria a reivindicação válida. O lance é **registrado, mas não executado** (o tabuleiro não muda).
- Se a posição atual já permite reivindicar, `legal_actions` inclui só `ClaimDraw()`, porque as versões com lance pretendido levariam ao mesmo empate.
- Um lance que dá mate não gera reivindicação.
- O estado após `ClaimDraw` é a mesma posição com `claimed_draw` preenchido: `is_terminal` vira `True` e o resultado é `DRAW / claimed_draw`.
- Texto: `ClaimDraw.uci()` produz `"claim"` ou `"claim:e2e4"`; `ClaimDraw.from_uci` faz o inverso.

**Custo:** para não testar todos os lances a cada nó, a verificação lance a lance só roda quando o contador de meios-lances é ≥ 7, o mínimo para uma repetição tripla. A detecção foi conferida contra `chess.Board.can_claim_draw()` em 300 partidas aleatórias (cerca de 99 mil posições), sem diferenças.

## 5. Fim de jogo

| Situação | Tipo | `reason` |
|---|---|---|
| Xeque-mate | automático | `checkmate` |
| Afogamento | automático | `stalemate` |
| Material insuficiente / posição morta detectada pela biblioteca | automático | `insufficient_material` |
| 75 lances sem captura ou movimento de peão | automático | `seventyfive_moves` |
| 5ª repetição | automático | `fivefold_repetition` |
| 3ª repetição ou 50 lances | **reivindicável** (`ClaimDraw`) | `claimed_draw` |

**Precedência:** o mate prevalece sobre os empates automáticos. Um mate no 150º meio-lance vale como mate.
**Limite documentado:** posições mortas usam apenas a detecção do `python-chess`; não há um solucionador geral.
Desistência, oferta de empate e relógio são do controlador da partida, não ações da árvore.

## 6. Contagem de transições

Cada chamada bem-sucedida a `next_state` soma 1 ao contador global, que mede o orçamento `max_transitions` da mesma forma para todos os algoritmos. Ações rejeitadas (`ValueError`) e `from_moves`/`deserialize` não contam.

```python
from game import reset_transition_count, transition_count
reset_transition_count()
action = agente.choose_action(state, budget)
gasto = transition_count()
```

## 7. Serialização

`serialize(state)` produz:

```json
{"start_fen": "...", "moves": ["e2e4", "e7e5"], "fen": "...",
 "claimed_draw": null, "result": "*", "reason": null}
```

Guardar só a FEN perde o histórico (e, com ele, as repetições). Por isso o formato guarda a FEN inicial e os lances. `deserialize` confere a FEN final e a reivindicação, e lança `ValueError` se algo não bater.

## 8. Ferramentas de depuração

```powershell
python -m game                          # partida aleatória animada no terminal
python -m game --delay 0 --seed 7       # só o tabuleiro final
python -m game --svg final.svg --pgn partida.pgn
python -m game --ascii                  # letras em vez de símbolos Unicode
python -m game.positions                # mostra o catálogo de posições
```

[`game/positions.py`](../game/positions.py) tem 20 posições funcionais (mates, afogamento, roques, *en passant*, promoções, repetições, 50 e 75 lances...) com o resultado esperado de cada uma: `positions.load("tripla_repeticao")`. Não é o conjunto experimental do Lichess.

## 9. Testes

```powershell
python -m pytest tests/test_chess_rules.py -q
```

Cobrem o contrato, a imutabilidade do estado, a contagem de transições, as reivindicações, todos os tipos de fim de jogo e a precedência do mate, roque, *en passant*, as quatro promoções, peça cravada, xeque, serialização (via JSON), PGN e o catálogo de posições.
