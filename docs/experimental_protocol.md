# Protocolo experimental (versão 2.0 — xadrez)

**Responsável:** Integrante 5 (Victor Alves) — `experiments/`
**Base:** `docs/PLANO_PROJETO_XADREZ_EDBII.md`, seções 8, 9, 11 e 13. Onde este arquivo e o plano divergirem, vale o plano.

Este documento diz **como** os experimentos são executados. Ele substitui o protocolo antigo de Pokémon.

---

## 1. Visão geral do fluxo

```
Lichess (PGN) ──► dataset.py ──► positions.csv (120 posições)
                                      │
        algoritmos (Flavio, Gustavo)  ▼
        ─────────────────────────► runner.py ──► decisions.csv   (tempo e memória)
                                      │
                       (depois, sem rodar junto)
                                      ▼
                              reference.py (Stockfish) ──► quality.csv
                                      │
                                      ▼
                              analysis.py ──► calibration_table.csv ──► Clara
                                          └─► gráficos 01–08 e empirical_summary.csv
```

Ordem de execução (plano, seção 14):

1. `dataset build` → 120 posições (60 calibração, 60 teste).
2. `runner decisions --set calibration` com **Minimax e GPN-MCTS** (a híbrida ainda não existe).
3. `reference evaluate` → qualidade da calibração.
4. `analysis` → `calibration_records.csv` → **entregar à Clara** (é o arquivo que o `load_calibration_csv` dela lê).
5. A híbrida lê o `calibration_records.csv`; então `runner decisions --set test ... --calibration-records experiments/results/calibration_records.csv` com os três agentes.
   **A calibração precisa cobrir os 6 orçamentos** × 6 contextos × 2 algoritmos: a híbrida dá erro se faltar qualquer combinação.
6. `reference evaluate` e `analysis` de novo; depois a rodada de **memória** e as **36 partidas**.

## 2. Posições (`experiments/dataset.py`)

- Fonte: Lichess Open Database, partidas padrão (`Variant` ausente ou `Standard`).
- 120 posições = 6 contextos × 20; cada contexto tem 10 de calibração e 10 de teste.
- Classificação (seção 9.1 do plano): **final** = no máximo 6 peças que não sejam reis ou peões; **abertura** = lance ≤ 10 e não final; **meio-jogo** = o resto. **Tática** = em xeque ou com captura legal.
- Excluídas: posições terminais, duplicadas ou com reivindicação de empate disponível.
- Divisão: cada **partida** é sorteada para calibração **ou** teste (hash com seed `2026`), então nenhuma partida aparece nos dois conjuntos.
- Cada posição guarda a **sequência de lances** (`moves_uci`) que leva até ela, para reconstruir o `chess.Board` com histórico. Não se usa só a FEN.
- `manifest.json` registra arquivo, SHA-256, data de obtenção, termos de reutilização, partidas lidas e seed.

## 3. Orçamentos e rodadas

| Tipo | Valores | Observação |
|---|---|---|
| Tempo por decisão (principal) | 100, 300, 1.000 ms | o cronômetro vale para toda a decisão |
| Transições por decisão (complementar) | 500, 2.000, 5.000 | |

- Um tipo de limite por vez (`Budget(max_time_ms=...)` **ou** `Budget(max_transitions=...)`).
- **Seeds:** 2026, 2027 e 2028 para GPN-MCTS e híbrida. Minimax não usa seed.
- **Rodada de tempo:** `time.perf_counter_ns()` e `time.process_time_ns()`, sem `tracemalloc`. Cada combinação roda `--reps` vezes (padrão 3).
- **Rodada de memória:** separada, com `tracemalloc`, 1 execução por combinação. O tempo dessa rodada **não** é usado.
- Cada decisão usa um **agente novo** e um **tabuleiro novo** (nada de árvore reaproveitada).

## 4. O que o runner confere em cada decisão

| Verificação | Resultado |
|---|---|
| O agente levantou exceção | `error` |
| O estado recebido foi modificado | `error` |
| A ação não está entre as legais | `illegal_action` |
| Tempo real maior que 2× o orçamento | `timeout` |
| Passou do orçamento (mas menos de 2×) | `ok` com `budget_overrun = True` |

Falhas são registradas como falhas, **nunca como empate**. Se um agente falha numa partida, a partida termina com resultado `*` e fica fora da pontuação.

## 5. Tabelas de resultados (`experiments/results/`)

| Arquivo | Uma linha é… |
|---|---|
| `decisions.csv` | uma decisão de um agente (posição ou partida) |
| `quality.csv` | a avaliação do Stockfish para uma decisão de `decisions.csv` |
| `games.csv` | uma partida completa |
| `pgn/*.pgn` | o PGN integral de cada partida |

Regras: cabeçalho fixo, gravação imediata, o experimento pode ser **interrompido e retomado** (combinações já gravadas são puladas), cada linha leva o `git_commit` (com `-dirty` se havia alteração não commitada). Arquivos brutos não são editados à mão.

Chaves esperadas em `Decision.extra` (conferidas com o código real do Gustavo e da Clara; falta o Flavio entregar o Minimax):

| Agente | Chaves |
|---|---|
| Todos | `tree_nodes_created` |
| Minimax | `alpha_beta_prunes`, `complete_depth` |
| GPN-MCTS (`GPNMCTSAgent`) | `iterations`, `playouts`, `playout_steps`, `tree_nodes` (vira `tree_nodes_created`). As colunas `proof_updates` e `pnrank_sorts` ficam vazias enquanto o agente não contar essas operações. `root_pn`, `root_visits`, `root_solved`, `time_s`, `fallback`, `formula` e `cpn` vão para `extra_json` |
| Híbrida (Clara) | `extra = {"hybrid": {...}, "agent_extra": {...}}`. O runner lê `hybrid.selected_algorithm`, `hybrid.context`, `hybrid.selection_time_ms`, `hybrid.probabilities` e junta o `agent_extra` (métricas do algoritmo escolhido) às colunas. O resto de `hybrid` vai para `extra_json` |

Qualquer outra chave vai para a coluna `extra_json`. Se o agente relatar `elapsed_ms`, `cpu_ms`, `peak_memory_kb`, `budget_overrun` ou `status`, esses valores ficam em `extra_json` dentro de `agent_reported`: **valem sempre as medidas do runner**, que cronometra de fora.

Forma de entrega aceita pelo runner: classe (`GPNMCTSAgent(seed=...)`; para o Minimax tenta `MinimaxAgent`, `Minimax` ou `AlphaBetaAgent`, ou a única classe do módulo com `choose_action`; a híbrida é `HybridAgent(minimax, gpn_mcts, tabela)`) **ou** função `choose_action(state, budget, *, seed=None, ...)` no módulo do algoritmo. A seed é passada em cada chamada (2026, 2027, 2028) por quem roda os experimentos.

## 6. Qualidade (`experiments/reference.py`)

- Stockfish 19, 1 thread, hash 32 MB, profundidade 14. O executável é registrado com nome e SHA-256.
- Melhor jogada: análise de todos os lances legais. Jogada do agente: análise restrita àquele lance (`root_moves`).
- **Top-1:** jogada do agente igual à melhor do Stockfish.
- **Perda:** `max(0, melhor − agente)` em centipeões, do ponto de vista de quem joga.
- **Mates** ficam fora da média em centipeões e são classificados à parte (`mate_kept`, `mate_missed`, `mate_blunder`, `lost_both`).
- O Stockfish roda **depois** dos experimentos, nunca junto (senão atrapalha as medidas de tempo).

## 7. Calibração para a Clara

`calibration_records.csv` (usado pela híbrida): uma linha por decisão, com `scenario_id, context, budget, algorithm, correct`. O `budget` usa os identificadores da híbrida (`100ms`, `300ms`, `1000ms`, `500_transitions`, `2000_transitions`, `5000_transitions`). A própria `load_calibration_csv` agrega por posição e aplica a suavização de Laplace.

`calibration_table.csv` traz a mesma informação em contagens, só para conferência, por algoritmo × contexto × orçamento:

- `tentativas` = número de posições de calibração do contexto;
- `acertos` = soma, sobre as posições, da **proporção** de acertos da posição.

Para o GPN-MCTS, as 3 seeds viram **uma** proporção por posição (cada posição pesa igual; as seeds não são 3 posições independentes). Por isso `acertos` pode ser fracionário. A Clara aplica `(acertos + 1) / (tentativas + 2)`.

A **política fixa** de referência é derivada em `analysis.derive_fixed_policy`: para cada orçamento, o algoritmo de maior concordância média na calibração (empate → Minimax), aplicado ao teste.

## 8. Partidas completas

- 3 confrontos × 12 partidas = 36: Minimax × GPN-MCTS, Híbrida × Minimax, Híbrida × GPN-MCTS.
- 6 partidas com cada agente de brancas; 300 ms por decisão; posição inicial padrão; sem livro.
- Seeds 2026, 2027, 2028 em rodízio. Pontuação: vitória 1, empate 0,5, derrota 0.
- Resultados descritivos, sem alegar significância estatística.

## 9. Gráficos e tabelas (`experiments/analysis.py`)

| # | Gráfico |
|---|---|
| 1 | Qualidade (top-1 e perda) × orçamento |
| 2 | Tempo × transições |
| 3 | Nós visitados e expandidos por algoritmo |
| 4 | Profundidade atingida |
| 5 | Memória |
| 6 | Qualidade por contexto (C1–C6) |
| 7 | Custo por iteração do GPN-MCTS |
| 8 | Resultados das partidas |

Também gera `empirical_summary.csv`, com a ramificação efetiva aproximada do Minimax (`nós^(1/profundidade completa)`). Ela serve para observar tendências, não para afirmar uma complexidade exata.

## 10. Divergência entre o plano e o código atual (decidir no grupo)

`algorithms/base.py` (branch da Clara) difere da seção 12.2 do plano:

| Plano 12.2 | `base.py` atual | O que o runner faz |
|---|---|---|
| `Budget.max_time_ms` | `Budget.max_time_s` (segundos) | aceita os dois; grava sempre em ms |
| `Decision.transitions` | não existe | usa `game.transition_count()` (zerado antes de cada decisão) |

Quando o grupo escolher um dos dois, nada precisa mudar nos experimentos.

## 10b. Pontos do plano que ficaram em aberto e como foram resolvidos aqui

Estes itens **não** estavam definidos no plano. Foram escolhidos para o código funcionar e precisam ser conhecidos pelo grupo.

| Ponto | Escolha | Onde mudar |
|---|---|---|
| Quando uma decisão vira `timeout` | tempo real > 2× o orçamento | `TIMEOUT_FACTOR` em `metrics.py` |
| Quantas repetições de medição de tempo | 3 | `DEFAULT_REPS` em `runner.py` |
| "Até o 10º lance completo" | número do lance ≤ 10 | `classify_phase` em `dataset.py` (a Clara deve usar a mesma regra) |
| Limite de tamanho da partida | 500 meios-lances; passou disso vira `max_plies` e não pontua | `MAX_PLIES` em `runner.py` |
| Uma posição por partida e contexto | só o candidato de menor hash de cada partida e contexto | `collect_candidates` em `dataset.py` |
| Agente na partida | uma instância por jogador por partida | `play_game` em `runner.py` |

## 11. Riscos que o grupo precisa conhecer

1. **Minimax com orçamento de tempo não é estritamente determinístico.** A profundidade concluída depende da velocidade da máquina naquele instante, então a ação pode mudar entre repetições. O runner registra isso em `action_changed`. Com orçamento de transições o resultado deve ser idêntico.
2. **Partidas repetidas.** Com seeds em rodízio (2026–2028) e posição inicial fixa, partidas com a mesma seed e os mesmos agentes tendem a ser iguais quando a busca é determinística. Conferir quantos PGNs distintos existem antes de interpretar as 12 partidas por confronto.
3. **Amostra pequena.** 20 posições por contexto e 10 por conjunto dão estimativas ruidosas; as conclusões por contexto devem ser descritas como tendência.
4. **Stockfish não é verdade absoluta.** É uma avaliação fixa em profundidade 14.

## 12. Dependências adicionais

`requirements.txt` do plano: `python-chess==1.999` e `pytest>=7`. Os experimentos precisam também de `pandas`, `matplotlib`, `numpy` e, só para ler arquivos `.pgn.zst`, `zstandard`. Ver `requirements-experiments.txt`.
