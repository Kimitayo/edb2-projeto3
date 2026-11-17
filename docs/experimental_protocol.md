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
4. `analysis` → `calibration_table.csv` → **entregar à Clara**.
5. Clara gera a tabela MEU; então `runner decisions --set test` com os três agentes.
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

Chaves esperadas em `Decision.extra` (combinar com Flavio, Gustavo e Clara):

| Agente | Chaves |
|---|---|
| Todos | `tree_nodes_created` |
| Minimax | `alpha_beta_prunes`, `complete_depth` |
| GPN-MCTS | `iterations`, `playouts`, `playout_steps`, `proof_updates`, `pnrank_sorts` |
| Híbrida | `chosen_algorithm` (`"minimax"` ou `"gpn_mcts"`), `context`, `selection_ms`, `p_minimax`, `p_gpn_mcts` |

Qualquer outra chave vai para a coluna `extra_json`.

## 6. Qualidade (`experiments/reference.py`)

- Stockfish 19, 1 thread, hash 32 MB, profundidade 14. O executável é registrado com nome e SHA-256.
- Melhor jogada: análise de todos os lances legais. Jogada do agente: análise restrita àquele lance (`root_moves`).
- **Top-1:** jogada do agente igual à melhor do Stockfish.
- **Perda:** `max(0, melhor − agente)` em centipeões, do ponto de vista de quem joga.
- **Mates** ficam fora da média em centipeões e são classificados à parte (`mate_kept`, `mate_missed`, `mate_blunder`, `lost_both`).
- O Stockfish roda **depois** dos experimentos, nunca junto (senão atrapalha as medidas de tempo).

## 7. Tabela de calibração para a Clara

`calibration_table.csv`, por algoritmo × contexto × orçamento:

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

## 10. Pontos do plano que ficaram em aberto e como foram resolvidos aqui

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
