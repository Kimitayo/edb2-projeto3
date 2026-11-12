# Protocolo Experimental

**Responsável:** Integrante 5 (Victor Alves) — Experimentos
**Status:** versão inicial (v0.1) — para o grupo revisar

> **Antes de ler:** o repositório `edb2-projeto3` ainda está só com a estrutura de pastas (`algorithms/`, `game/`, `tests/`, `paper/`, `docs/`) e o README vazio. Por isso este protocolo é **genérico**: usa `ALG_A`, `ALG_B`... no lugar dos algoritmos reais e números iniciais que o grupo deve ajustar. Os trechos marcados com **[DECIDIR]** precisam da opinião do grupo.

---

## 1. Objetivo

Comparar o desempenho de diferentes algoritmos que escolhem ações em batalhas, medindo **qualidade da decisão** (ganhar a batalha) e **custo computacional** (nós, tempo, memória), com dados que possam ser **reproduzidos** por qualquer integrante.

Perguntas que os experimentos devem responder:

1. Com o mesmo orçamento, qual algoritmo vence mais?
2. Quanto o desempenho melhora quando o orçamento aumenta?
3. Qual algoritmo gasta menos tempo/memória para o mesmo nível de qualidade?
4. O resultado muda quando o cenário fica maior ou mais difícil?

---

## 2. Métricas registradas

| Grupo | Métrica | Como medir | Coluna no CSV |
|---|---|---|---|
| Qualidade | Vencedor da batalha | `A`, `B` ou `draw` | `winner` |
| Qualidade | Nº de turnos | contagem na simulação | `turns` |
| Qualidade | Vida restante de cada lado | ao fim da batalha | `hp_remaining_a/b` |
| Custo | Nós visitados | `SearchCounter.visit()` | `nodes_visited` |
| Custo | Nós expandidos | `SearchCounter.expand()` | `nodes_expanded` |
| Custo | Profundidade máxima | maior `depth` visto | `max_depth` |
| Custo | Tempo total e médio por decisão | `time.perf_counter_ns()` | `total_time_ms`, `mean_decision_time_ms` |
| Custo | Pico de memória | `tracemalloc` | `peak_memory_kb` |
| Controle | Vezes que o orçamento acabou | `counter.exhausted()` | `budget_exhausted_count` |

**Taxa de vitória** e **intervalos de confiança** não são colunas: são calculados depois, na análise.

---

## 3. Como contar nós visitados e expandidos

Definição única, valendo para **todos** os algoritmos:

- **Nó visitado:** todo estado em que o algoritmo "entra" para avaliar (inclui folhas e avaliações por heurística ou simulação).
- **Nó expandido:** todo estado para o qual o algoritmo **gerou os sucessores** (os filhos).

Logo, `expandidos ≤ visitados` (o código de gravação recusa linhas que violem isso).

**Regra de ouro:** o algoritmo não conta por conta própria. Ele recebe um `SearchCounter` e chama `counter.visit(depth)` e `counter.expand()`. Assim a contagem é igual para todos.

> **[DECIDIR]** Em algoritmos com simulação aleatória (ex.: playouts), cada passo simulado conta como nó visitado? Sugestão: **sim**, para a comparação ser justa.

---

## 4. Orçamentos computacionais

O orçamento vale **por decisão** (por jogada), não por batalha.

| Tipo | Valores iniciais sugeridos | Uso |
|---|---|---|
| **Nós** (principal) | 100 · 500 · 1.000 · 5.000 · 10.000 | Comparação justa e reproduzível (não depende da máquina) |
| **Tempo** (secundário) | 10 ms · 50 ms · 200 ms | Mostra o que acontece "no mundo real" |

Regras:

- Todos os algoritmos de um mesmo experimento usam **o mesmo orçamento**.
- Quando o orçamento acaba, o algoritmo devolve a **melhor ação encontrada até ali** (e `budget_exhausted_count` aumenta).
- Os valores acima são um ponto de partida: rodar um teste-piloto e ajustar para que o menor orçamento ainda "funcione" e o maior não demore horas.

---

## 5. Cenários de batalha

Cada cenário é gerado por uma **semente (seed)**: mesma semente = mesmo cenário, sempre.

**Parâmetros que variam [DECIDIR conforme o jogo]:**

- número de unidades por lado
- vida e ataque das unidades
- tamanho do mapa / número de ações possíveis por turno (fator de ramificação)
- presença de aleatoriedade no jogo (sim/não)

**Níveis de dificuldade (sugestão):** `small`, `medium`, `large`.

**Procedimento:**

1. Escolher uma semente-base fixa (ex.: `2026`).
2. Gerar `N = 100` cenários por nível: `scenario_seed = base + índice`.
3. Salvar a lista de cenários (seeds) em arquivo **antes** de rodar qualquer algoritmo.
4. Todos os algoritmos jogam **exatamente os mesmos cenários**.
5. Trocar de lado (quem começa) em metade das partidas, para não favorecer ninguém.

---

## 6. Medição de tempo e memória

**Tempo**
- `time.perf_counter_ns()` para tempo real; `time.process_time_ns()` para tempo de CPU.
- Medido **por decisão** e somado por batalha.

**Memória**
- `tracemalloc` (pico de memória alocada pelo Python durante a decisão).
- O `tracemalloc` **deixa o código mais lento**, então o protocolo usa **duas rodadas separadas**:
  - **Rodada T (tempo):** sem `tracemalloc` → preenche os campos de tempo.
  - **Rodada M (memória):** com `tracemalloc` → preenche `peak_memory_kb`; o tempo desta rodada é **descartado**.

**Cuidados para o tempo ser confiável**
- Fazer **aquecimento** (descartar as 3 primeiras batalhas).
- Rodar sempre na **mesma máquina**, com poucos programas abertos e na tomada (se notebook).
- Repetir cada configuração **R = 5 vezes** (com seeds de execução diferentes) e usar mediana/média + desvio.
- Anotar a máquina usada (CPU, RAM, versão do Python) em `docs/ambiente.md`.

---

## 7. Armazenamento dos resultados (CSV)

- Arquivos em `experiments/results/`, um por experimento (ex.: `exp01_orcamento_nos.csv`).
- **Uma linha = uma batalha de um algoritmo em um cenário.**
- Colunas fixas definidas em `experiments/results_storage.py` (`RunRecord`): identificação (`experiment_id`, `scenario_seed`, `run_seed`, `git_commit`...), configuração (`algorithm`, `budget_type`, `budget_value`...), resultados e controle (`status`).
- Cada linha é gravada na hora (`flush`), então uma falha não perde o que já rodou.
- Arquivos brutos **nunca** são editados à mão; limpeza e agregação acontecem em scripts de análise separados.
- Registrar o `git_commit` de cada execução, para saber exatamente qual código gerou o dado.

---

## 8. Experimentos planejados

| ID | Pergunta | Variável que muda | Fixo |
|---|---|---|---|
| `exp01_orcamento_nos` | Como o desempenho cresce com o orçamento? | orçamento de nós | cenários `medium`, todos os algoritmos |
| `exp02_orcamento_tempo` | E com orçamento de tempo? | orçamento de tempo | idem |
| `exp03_escalabilidade` | O que muda com cenários maiores? | `small/medium/large` | orçamento médio |
| `exp04_memoria` | Quanto cada algoritmo consome? | algoritmo | Rodada M |

---

## 9. Gráficos planejados para o artigo

1. **Taxa de vitória × orçamento (nós)** — linhas por algoritmo, com barras de erro. *(exp01)*
2. **Tempo médio por decisão × orçamento** — mostra o custo real. *(exp01/02)*
3. **Taxa de vitória × tempo médio** (fronteira qualidade-custo) — o gráfico mais "conclusivo".
4. **Nós visitados vs. expandidos** por algoritmo — barras agrupadas.
5. **Pico de memória por algoritmo** — barras. *(exp04)*
6. **Desempenho por nível de dificuldade** — barras agrupadas. *(exp03)*

Ferramenta sugerida: `pandas` + `matplotlib` (confirmar no `requirements.txt`).

---

## 10. Ameaças à validade (o que pode dar errado)

- Máquina ocupada durante os testes → tempo ruidoso. *Mitigação:* repetições + mediana.
- Algoritmos contando nós de formas diferentes. *Mitigação:* `SearchCounter` único (seção 3).
- Poucos cenários → resultado por sorte. *Mitigação:* N ≥ 100 por nível.
- Código mudando entre experimentos. *Mitigação:* gravar `git_commit`.

---

## 11. Esboço da interface (adiado)

Não será desenvolvida agora. Ideia para depois: tela com o tabuleiro/batalha, escolha de algoritmo e orçamento, botão "rodar", e painel com nós visitados, tempo e memória em tempo real.

---

## 12. Perguntas abertas para o grupo

1. Quais são os algoritmos finais e em qual pasta/arquivo cada um está?
2. Qual é a interface esperada dos algoritmos? (sugestão: `choose_action(state, counter)`)
3. O jogo tem aleatoriedade? Se sim, como controlar a seed dentro dele?
4. Os orçamentos da seção 4 são viáveis no tempo que temos?

---

## 13. Entregas desta etapa

- [x] `experimental_protocol.md` (este arquivo)
- [x] `experiments/metrics.py` — `Budget`, `SearchCounter`, `measure()`, `BattleAggregator`
- [x] `experiments/results_storage.py` — `RunRecord`, `ResultsWriter`, `load_results`
- [x] `experiments/test_metrics.py` — testes rápidos (passando)
- [ ] Gerador de cenários (depende do jogo, em `game/`)
- [ ] Script *runner* que liga algoritmos + métricas + CSV
- [ ] Scripts de gráficos
