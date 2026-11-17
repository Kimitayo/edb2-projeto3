# Projeto EDB II — Busca em Árvores de Jogo no Xadrez

**Especificação oficial do grupo — versão 2.0**  
**Repositório:** https://github.com/Kimitayo/edb2-projeto3  
**Data:** 16/11/2026

> **Padrão do grupo:** todas as definições deste documento estão adotadas. Elas devem ser utilizadas por todos os integrantes na implementação e nos experimentos. Não há decisões técnicas aguardando aprovação. Os checklists representam somente atividades que precisam ser executadas.

---

# 1. Projeto

## 1.1. Tema

**Comparação entre Minimax com poda Alfa-Beta, GPN-MCTS e uma busca híbrida contextual baseada em Máxima Utilidade Esperada em partidas de xadrez.**

## 1.2. Objetivo

Implementar e comparar duas estratégias de busca em árvores de jogo — Minimax com Alfa-Beta e Generalized Proof-Number Monte-Carlo Tree Search (GPN-MCTS) — e desenvolver uma terceira configuração híbrida que escolha entre elas de acordo com o contexto da posição.

A comparação considerará:

- qualidade das decisões;
- custo computacional;
- desempenho em partidas;
- complexidade temporal e espacial;
- comportamento em diferentes fases e situações do jogo.

## 1.3. Pergunta de pesquisa

> Uma política contextual baseada em Máxima Utilidade Esperada consegue selecionar entre Minimax com Alfa-Beta e GPN-MCTS e melhorar a qualidade das decisões de xadrez sob um orçamento computacional igual?

## 1.4. Contribuição

A contribuição do grupo será implementar os métodos sob as mesmas regras e interfaces, comparar seu comportamento em xadrez padrão e avaliar uma política híbrida que seleciona o algoritmo de acordo com resultados experimentais anteriores.

O trabalho não afirmará que inventou Minimax, Alfa-Beta ou GPN-MCTS. A contribuição estará na comparação, na análise de complexidade e na aplicação da seleção contextual por MEU.

A estrutura analisada é uma **árvore de busca adversarial**:

- **nó:** estado completo de uma partida;
- **aresta:** movimento legal que leva a outro estado;
- **raiz:** posição entregue ao algoritmo;
- **ramificação:** movimentos legais disponíveis em um estado;
- **profundidade:** quantidade de meios-lances (*plies*);
- **folha:** estado terminal ou posição em que a busca foi interrompida;
- **expansão:** geração dos estados sucessores.

---

# 2. Padrão técnico único

| Componente | Padrão oficial |
|---|---|
| Jogo | Xadrez tradicional |
| Regras | Todas as regras oficiais de movimentação, vitória e empate |
| Biblioteca | `python-chess` |
| Linguagem | Python 3.12 |
| Estado | `chess.Board`, preservando o histórico |
| Ação | `chess.Move` ou ação especial de reivindicação de empate |
| Busca tradicional | Minimax com poda Alfa-Beta |
| Busca recente | GPN-MCTS |
| Busca híbrida | Seleção contextual por Máxima Utilidade Esperada |
| Utilidade terminal | Vitória `+1`, empate `0`, derrota `−1` |
| Heurística | Função material e posicional da Seção 5 |
| Orçamento principal | Tempo igual por decisão: 100 ms, 300 ms e 1.000 ms |
| Orçamento complementar | Transições iguais por decisão: 500, 2.000 e 5.000 |
| Referência de qualidade | Stockfish 19, profundidade 14 |
| Dados | Partidas padrão públicas do Lichess |
| Teste-piloto para ajuste | Não será realizado |
| Validação funcional | Obrigatória para regras, algoritmos e integração |

Dependências do projeto:

```text
python-chess==1.999
pytest>=7
```

Todos os integrantes utilizarão Python 3.12 e as mesmas versões registradas no arquivo de dependências do repositório.

---

# 3. Xadrez: regras, estado e ações

**Responsável pelo motor:** Matheus Bulhões.

## 3.1. Regras utilizadas

O simulador utilizará xadrez padrão, incluindo:

- tabuleiro 8 × 8 e peças tradicionais;
- movimentação legal das seis classes de peças;
- capturas;
- roque pequeno e grande;
- captura *en passant*;
- promoção para dama, torre, bispo ou cavalo;
- proibição de movimentos que deixem o próprio rei em xeque;
- xeque e xeque-mate;
- afogamento;
- empate por posição morta/insuficiência de material nos casos detectados pela biblioteca;
- repetição de posição;
- regra dos 50 movimentos, quando reivindicada;
- regra dos 75 movimentos, quando automática.

### Empates reivindicáveis e automáticos

A implementação respeitará a diferença entre:

- **Reivindicáveis:** terceira repetição e regra dos 50 movimentos.
- **Automáticos:** quinta repetição e regra dos 75 movimentos, respeitada a precedência do xeque-mate.

Uma reivindicação válida será representada por uma ação especial `CLAIM_DRAW`. Quando a reivindicação depender de um movimento pretendido, esse movimento será registrado, mas não executado.

Desistência, oferta de empate e relógio de torneio pertencem ao controlador da partida e não serão ações da árvore. O relógio de xadrez não será simulado; os limites de tempo deste projeto são orçamentos de processamento dos algoritmos.

**Limite documentado:** será utilizada a detecção de posição morta/insuficiência de material fornecida pelo `python-chess`. O grupo não implementará um solucionador geral de todas as posições mortas excepcionais.

## 3.2. Representação do estado

Cada estado será representado por `chess.Board`, mantendo o histórico de movimentos.

O estado deverá preservar:

- localização, tipo e cor das peças;
- jogador da vez;
- direitos de roque;
- possibilidade de captura *en passant*;
- contador de meios-lances desde captura ou movimento de peão;
- número do lance;
- histórico necessário à verificação de repetição.

Para simular uma jogada, o algoritmo deverá criar uma cópia com `copy(stack=True)` e aplicar o movimento na cópia. O estado recebido nunca poderá ser modificado.

FEN isolada não será usada como representação completa do histórico. As posições experimentais guardarão PGN ou a sequência de movimentos que levou à posição.

## 3.3. Ações e interface do motor

Ações:

```python
Action = chess.Move | ClaimDraw
```

Funções comuns:

| Função | Responsabilidade |
|---|---|
| `legal_actions(state)` | Retornar movimentos legais e reivindicações válidas |
| `next_state(state, action)` | Criar o estado sucessor sem modificar o original |
| `is_terminal(state)` | Identificar encerramento automático ou reivindicação aceita |
| `get_outcome(state)` | Retornar vitória das brancas, vitória das pretas, empate ou andamento |
| `utility(state, player)` | Retornar `+1`, `0` ou `−1` em estados terminais |
| `evaluate(state, player)` | Calcular a heurística da Seção 5 |
| `state_features(state)` | Extrair as características usadas pela híbrida |
| `render(state)` | Exibir o tabuleiro para depuração |
| `serialize(state)` | Registrar estado e histórico |

O motor validará a legalidade antes de executar qualquer movimento. Todos os algoritmos utilizarão exatamente essa mesma interface.

---

# 4. Utilidade dos estados terminais

A utilidade será calculada do ponto de vista do jogador avaliado:

\[
U(s,p)=
\begin{cases}
+1 & \text{se o jogador }p\text{ venceu}\\
0 & \text{se houve empate}\\
-1 & \text{se o jogador }p\text{ perdeu}
\end{cases}
\]

A utilidade é igual para Minimax, GPN-MCTS e híbrida.

Uma vitória ou derrota comprovada sempre prevalece sobre qualquer valor heurístico. A heurística será usada somente em estados não terminais.

---

# 5. Heurística fixa do projeto

**Responsável pela implementação compartilhada:** Flavio Neves.  
**Utilização:** Minimax e avaliações não terminais dos playouts do GPN-MCTS.

A heurística considera material e características posicionais. Os valores abaixo são fixos para todos os algoritmos e experimentos.

## 5.1. Valores materiais

| Peça | Valor em centipeões |
|---|---:|
| Peão | 100 |
| Cavalo | 320 |
| Bispo | 330 |
| Torre | 500 |
| Dama | 900 |
| Rei | Não recebe valor material |

## 5.2. Características posicionais

| Característica | Peso | Regra de contagem |
|---|---:|---|
| Mobilidade | `+5` | Casas atacadas por cavalos, bispos, torres e damas, excluindo casas ocupadas por peças próprias; contagem por peça |
| Controle do centro | `+12` | Cada casa entre `d4`, `e4`, `d5` e `e5` atacada pelo jogador, contada uma vez |
| Peões dobrados | `−20` | Cada peão adicional na mesma coluna |
| Peões isolados | `−15` | Cada peão sem peões aliados nas colunas adjacentes |
| Peões passados | `+30` | Cada peão sem peões adversários à frente na mesma coluna ou nas adjacentes |
| Par de bispos | `+30` | Bônus único quando o jogador possui pelo menos dois bispos |
| Risco do rei | `−20` | Cada casa adjacente ao rei atacada pelo adversário |
| Rei em xeque | `−40` | Penalidade adicional quando o rei do jogador está em xeque |

A mobilidade será calculada usando os ataques pseudo-legais das peças (`attacks()`), sem excluir peças cravadas. Essa regra torna a avaliação determinística e igual para todos os agentes.

Para cada característica:

\[
\Delta f=f(\text{jogador avaliado})-f(\text{adversário})
\]

Definições:

- \(P,N,B,R,Q\): quantidade de peões, cavalos, bispos, torres e damas;
- \(Mob\): mobilidade;
- \(Centro\): casas centrais atacadas;
- \(Dob\): peões dobrados adicionais;
- \(Iso\): peões isolados;
- \(Pass\): peões passados;
- \(PB\): presença do par de bispos (`1` ou `0`);
- \(RiscoRei\): casas adjacentes ao rei atacadas pelo adversário;
- \(EmXeque\): vale `1` se o rei está em xeque, caso contrário `0`.

## 5.3. Fórmula

\[
\begin{aligned}
S(s,p)={}&100\Delta P+320\Delta N+330\Delta B\\
&+500\Delta R+900\Delta Q\\
&+5\Delta Mob+12\Delta Centro\\
&-20\Delta Dob-15\Delta Iso\\
&+30\Delta Pass+30\Delta PB\\
&-20\Delta RiscoRei-40\Delta EmXeque
\end{aligned}
\]

Normalização:

\[
\boxed{
H(s,p)=0{,}99\tanh\left(\frac{S(s,p)}{600}\right)
}
\]

### Regras obrigatórias

1. \(H(s,\text{brancas})=-H(s,\text{pretas})\).
2. A avaliação não terminal fica estritamente entre `−0,99` e `+0,99`.
3. A posição inicial deve ter avaliação zero.
4. A mesma posição deve receber o mesmo valor nos três agentes.
5. Mate comprovado utiliza a utilidade terminal, não a heurística.
6. Os pesos não serão ajustados após a execução dos experimentos finais.

Os pesos são parâmetros fixos escolhidos para este experimento; não serão descritos como valores ótimos para xadrez.

---

# 6. Algoritmo 1 — Minimax com Alfa-Beta

**Responsável:** Flavio Neves.

## 6.1. Configuração

- Minimax com alternância MAX/MIN.
- Poda Alfa-Beta.
- Aprofundamento iterativo nas profundidades 1, 2, 3 e seguintes, até o orçamento acabar.
- Utilidade terminal \(U\).
- Heurística compartilhada \(H\) nas folhas não terminais.
- Sem tabela de transposição na configuração principal.
- Ordenação determinística dos movimentos:
  1. capturas;
  2. promoções;
  3. xeques;
  4. demais movimentos;
  5. desempate pela notação UCI.
- Retorno da melhor ação da última profundidade completamente concluída.
- Se o orçamento terminar antes de concluir a primeira profundidade, retornar a primeira ação legal em ordem UCI.

## 6.2. Métricas próprias

- nós visitados;
- nós expandidos;
- transições;
- profundidade máxima;
- profundidade completa;
- quantidade de podas Alfa-Beta;
- tempo real;
- tempo de CPU;
- memória;
- ação selecionada.

## 6.3. Checklist de Flavio

- [ ] Implementar a heurística completa da Seção 5.
- [ ] Implementar Minimax.
- [ ] Implementar poda Alfa-Beta.
- [ ] Implementar aprofundamento iterativo.
- [ ] Implementar ordenação determinística.
- [ ] Implementar interrupção por orçamento.
- [ ] Retornar a melhor jogada da última profundidade completa.
- [ ] Garantir movimento legal e estado original intacto.
- [ ] Testar mate em um e em dois lances.
- [ ] Comparar Alfa-Beta com Minimax sem poda em árvores pequenas.
- [ ] Registrar todas as métricas próprias.
- [ ] Escrever a análise de complexidade.

**Estudar:** árvores de jogo, MAX/MIN, Minimax, Alfa-Beta, profundidade, fator de ramificação, ordenação de movimentos, heurísticas e análise assintótica.

**Entrega:** `algorithms/minimax.py`, `game/evaluation.py`, testes e análise de complexidade.

---

# 7. Algoritmo 2 — GPN-MCTS

**Responsável:** Gustavo.

## 7.1. Base científica

Kowalski, J.; Soemers, D. J. N. J.; Kosakowski, S.; Winands, M. H. M. (2025). *Generalized Proof-Number Monte-Carlo Tree Search*. ECAI 2025. DOI: [10.3233/FAIA251406](https://doi.org/10.3233/FAIA251406).

O GPN-MCTS combina MCTS com números de prova por jogador para direcionar a seleção da árvore a regiões com maior possibilidade de serem comprovadas. O artigo também apresenta diferentes vieses baseados em números de prova e integração com Score-Bounded MCTS.

## 7.2. Variante fixa do grupo

| Componente | Configuração |
|---|---|
| Busca | GPN-MCTS |
| Seleção | UCT-PN |
| Viés | PNRank |
| Constante UCT | \(C=\sqrt{2}\) |
| Constante do viés | \(C_{pn}=1{,}0\) |
| Provas | Um número de prova por jogador |
| Score-Bounded MCTS | Habilitado |
| Reutilização da árvore entre decisões | Desabilitada |
| Playout | Movimentos legais pseudoaleatórios |
| Limite do playout | 80 meios-lances |
| Seeds | 2026, 2027 e 2028 |
| Corte não terminal | Utiliza \(H\) como estimativa, nunca como prova |
| Escolha final | Filho com mais visitas; empate por UCI |

A escolha de PNRank e \(C_{pn}=1{,}0\) é o padrão fixo deste projeto. O artigo mostra que não existe um valor de \(C_{pn}\) universalmente melhor para todos os jogos. Portanto, o grupo não afirmará que esse parâmetro é ótimo para xadrez.

A implementação sem reutilização de árvore entre decisões é uma adaptação experimental para que cada posição seja pesquisada do zero sob o mesmo orçamento. Essa diferença em relação a configurações do artigo será descrita no trabalho.

## 7.3. Números de prova

Cada nó mantém um número de prova para cada jogador.

- Nó desconhecido: número inicial `1`.
- Vitória terminal comprovada para o jogador: prova `0`.
- Resultado terminal que não representa vitória para aquele jogador: prova infinita.
- Em nó OR, a prova considera o menor valor entre os filhos.
- Em nó AND, a prova considera a soma dos valores dos filhos.

A atualização deverá seguir os papéis OR/AND e a definição por jogador do artigo original.

**Regra essencial:** uma avaliação heurística não comprova vitória ou derrota. Somente resultado terminal ou limite exato demonstrado pode atualizar números de prova como resolvidos.

## 7.4. Seleção e playout

A seleção UCT-PN utilizará:

\[
UCT\text{-}PN =
\overline{X}_i+
C\sqrt{\frac{\ln N}{n_i}}+
C_{pn}\cdot PNRank(i)
\]

O PNRank ordena os filhos segundo os números de prova; o melhor recebe o maior viés. Empates recebem o mesmo rank, normalizado no intervalo `[0,1]`.

No playout:

- selecionar movimentos legais pseudoaleatórios;
- parar em estado terminal ou após 80 meios-lances;
- utilizar \(U\) se o estado for terminal;
- utilizar \(H\) se atingir o limite sem resultado terminal;
- não transformar valor heurístico em prova.

O Score-Bounded MCTS utilizará apenas limites de pontuação válidos. Estimativas heurísticas de corte não serão tratadas como limites comprovados.

## 7.5. Métricas próprias

- iterações;
- playouts;
- passos de playout;
- nós visitados e expandidos;
- nós criados;
- transições;
- profundidade máxima;
- atualizações de números de prova;
- operações de ordenação PNRank;
- tempo real e CPU;
- memória;
- ação selecionada.

## 7.6. Checklist de Gustavo

- [ ] Estudar o artigo GPN-MCTS de 2025.
- [ ] Estudar MCTS e UCT.
- [ ] Estudar Proof-Number Search e árvores AND/OR.
- [ ] Implementar nós e filhos.
- [ ] Implementar visitas, recompensas e provas por jogador.
- [ ] Implementar propagação OR/AND.
- [ ] Implementar PNRank.
- [ ] Implementar UCT-PN com os parâmetros fixos.
- [ ] Implementar expansão e playout de até 80 meios-lances.
- [ ] Integrar Score-Bounded MCTS.
- [ ] Garantir que heurística não gere prova.
- [ ] Implementar seleção final do movimento.
- [ ] Testar provas e refutações em árvores pequenas.
- [ ] Garantir ação legal e reprodutibilidade por seed.
- [ ] Registrar métricas e escrever a análise de complexidade.

**Estudar:** MCTS, UCT, Proof-Number Search, números de prova, AND/OR, PNRank, Score-Bounded MCTS, playout e retropropagação.

**Entrega:** `algorithms/gpn_mcts.py`, testes, documentação da variante e análise de complexidade.

---

# 8. Orçamento e comparação justa

A comparação principal utilizará **o mesmo tempo real por decisão** para Minimax e GPN-MCTS. Isso considera o custo de todo o trabalho interno de cada algoritmo, inclusive a manutenção dos números de prova.

## 8.1. Experimento principal: tempo igual

| Nível | Tempo por decisão |
|---|---:|
| Curto | 100 ms |
| Principal | 300 ms |
| Ampliado | 1.000 ms |

O cronômetro começa antes da extração/ordenação necessária à busca e termina quando a ação é retornada.

Para a híbrida, o cronômetro inclui extração de características, consulta à tabela MEU e execução do algoritmo selecionado.

## 8.2. Experimento complementar: transições iguais

| Nível | Transições por decisão |
|---|---:|
| Curto | 500 |
| Principal | 2.000 |
| Ampliado | 5.000 |

Uma transição é a aplicação de um movimento para produzir um estado sucessor. Todas as transições contam, inclusive as realizadas nos playouts.

A comparação por transições é complementar: uma transição de Minimax e uma de GPN-MCTS podem exigir quantidades diferentes de trabalho interno. Por isso, o tempo igual é a comparação principal de custo computacional.

## 8.3. Regras de execução

- Mesma máquina, sistema operacional, Python e dependências.
- Mesmo estado, histórico, jogador da vez e movimentos legais.
- Nenhum algoritmo consulta Stockfish durante sua busca.
- Nenhum algoritmo reutiliza árvore de uma decisão anterior.
- GPN-MCTS utiliza seeds 2026, 2027 e 2028.
- Minimax é determinístico; repetições de medição devem retornar a mesma ação.
- Tempo real medido com `time.perf_counter_ns()`.
- Tempo de CPU medido com `time.process_time_ns()`.
- Memória medida em execução separada com `tracemalloc`, pois a instrumentação altera o tempo.
- Falhas, timeouts e movimentos ilegais são registrados como falhas, nunca como empate.
- Se o orçamento terminar, o agente retorna a melhor ação legal já encontrada.

## 8.4. Métricas comuns

| Métrica | Definição |
|---|---|
| `nodes_visited` | Estados visitados ou avaliados |
| `nodes_expanded` | Estados cujos sucessores foram gerados |
| `tree_nodes_created` | Nós armazenados na árvore |
| `transitions` | Movimentos efetivamente aplicados |
| `max_depth_reached` | Maior profundidade em meios-lances |
| `elapsed_ms` | Tempo real total |
| `cpu_ms` | Tempo de CPU |
| `peak_memory_kb` | Pico de memória medido em rodada separada |
| `budget_overrun` | Indica ultrapassagem do orçamento |
| `status` | `ok`, `timeout`, `illegal_action` ou `error` |

Métricas específicas:

- Minimax: podas Alfa-Beta e profundidade completa.
- GPN-MCTS: iterações, playouts, atualizações de prova e ordenações PNRank.
- Híbrida: contexto, probabilidades, algoritmo escolhido e custo de seleção.

Nós visitados, nós expandidos e transições serão apresentados separadamente; eles não serão tratados como unidades equivalentes de trabalho.

---

# 9. Dados, referência e avaliação da qualidade

**Responsável pela experimentação:** Victor Alves.

## 9.1. Fonte e conjunto de posições

Fonte: [Lichess Open Database — partidas padrão](https://database.lichess.org/).

Serão selecionadas 120 posições legais de partidas padrão públicas, com registro do PGN de origem e da sequência de movimentos. A fonte, a data de obtenção, os identificadores e os termos de reutilização serão registrados no manifesto dos dados.

As posições serão divididas em seis contextos, com 20 posições por contexto:

| Fase | Situação | Quantidade |
|---|---|---:|
| Abertura | Não tática | 20 |
| Abertura | Tática | 20 |
| Meio-jogo | Não tática | 20 |
| Meio-jogo | Tática | 20 |
| Final | Não tática | 20 |
| Final | Tática | 20 |
| **Total** | | **120** |

### Classificação da fase

- **Abertura:** até o 10º lance completo, desde que não seja final.
- **Final:** no máximo seis peças que não sejam reis ou peões, somadas entre os dois lados.
- **Meio-jogo:** posições que não se enquadrem como abertura ou final.

### Classificação tática

- **Tática:** o jogador está em xeque ou possui ao menos uma captura legal.
- **Não tática:** não está em xeque e não possui captura legal.

Essas são definições operacionais fixas do experimento, não definições universais das fases do xadrez.

## 9.2. Calibração e teste

Cada contexto terá:

- 10 posições para calibração da híbrida;
- 10 posições para avaliação final.

| Conjunto | Quantidade |
|---|---:|
| Calibração | 60 |
| Teste final | 60 |
| Total | 120 |

Posições da mesma partida não poderão aparecer nos dois conjuntos. A separação será feita deterministicamente com seed `2026`.

Posições já terminais, duplicadas ou com reivindicação de empate disponível serão excluídas da avaliação principal. Elas continuarão presentes nos testes funcionais do motor.

## 9.3. Referência externa

O Stockfish 19 será utilizado exclusivamente para avaliar a qualidade das jogadas, nunca para escolher jogadas durante a execução dos agentes.

| Configuração | Valor |
|---|---|
| Motor | Stockfish 19 |
| Threads | 1 |
| Hash | 32 MB |
| Profundidade | 14 |
| Livro de abertura | Desabilitado |

A versão e o hash do executável serão registrados.

A referência é uma avaliação computacional fixa, não uma prova matemática de jogada ótima.

## 9.4. Métricas de qualidade

1. **Concordância top-1:** proporção de jogadas idênticas à melhor jogada indicada pelo Stockfish.
2. **Perda em centipeões:** diferença entre a avaliação da melhor jogada de referência e a avaliação da jogada escolhida pelo agente.
3. **Resultado competitivo:** vitórias, empates e derrotas nas partidas completas.

Para a perda em centipeões, o Stockfish analisará a posição na profundidade 14. A melhor jogada será avaliada considerando todos os movimentos legais; a jogada do agente será avaliada restringindo a análise àquela jogada. A perda será:

\[
Regret=\max(0,\ Score_{melhor}-Score_{agente})
\]

Os escores serão expressos do ponto de vista do jogador que está para mover. Resultados de mate serão tratados separadamente como resultados decisivos, acima dos escores em centipeões.

## 9.5. Partidas completas

Serão realizadas 12 partidas por confronto:

- Minimax × GPN-MCTS;
- Híbrida × Minimax;
- Híbrida × GPN-MCTS.

Total: **36 partidas**.

Configuração:

- seis partidas com cada agente jogando de brancas;
- orçamento de 300 ms por decisão;
- posição inicial padrão;
- sem livro de abertura;
- seeds registradas para GPN-MCTS;
- PGN integral de cada partida salvo;
- vitória = 1 ponto, empate = 0,5 e derrota = 0.

Os resultados das 12 partidas por confronto serão apresentados como avaliação competitiva complementar e descritiva, sem alegar significância estatística forte a partir de uma amostra pequena.

---

# 10. Busca híbrida por Máxima Utilidade Esperada

**Responsável:** Clara.

## 10.1. Funcionamento

A híbrida recebe uma posição, identifica seu contexto e escolhe **um único algoritmo** para realizar a busca:

- Minimax com Alfa-Beta; ou
- GPN-MCTS.

A híbrida não executa as duas buscas completas para depois escolher a resposta. A seleção acontece antes da busca principal.

## 10.2. Contextos

A política utiliza duas características:

1. fase: abertura, meio-jogo ou final;
2. situação: tática ou não tática.

São seis contextos:

| Código | Fase | Situação |
|---|---|---|
| C1 | Abertura | Não tática |
| C2 | Abertura | Tática |
| C3 | Meio-jogo | Não tática |
| C4 | Meio-jogo | Tática |
| C5 | Final | Não tática |
| C6 | Final | Tática |

A política também registra fator de ramificação, diferença de material e situação de xeque, mas não cria subdivisões adicionais.

## 10.3. Crenças e probabilidades

A calibração mede a probabilidade de cada algoritmo escolher a mesma jogada que o Stockfish.

\[
Q=
\begin{cases}
1 & \text{se a jogada coincide com a referência}\\
0 & \text{caso contrário}
\end{cases}
\]

Para algoritmo \(a\), contexto \(x\) e orçamento \(B\):

\[
\boxed{
P(Q=1\mid a,x,B)=
\frac{acertos_{a,x,B}+1}{tentativas_{a,x,B}+2}
}
\]

A suavização de Laplace evita probabilidades exatamente iguais a zero ou um.

Para GPN-MCTS, as três seeds serão executadas em cada posição. A proporção de acertos será calculada primeiro por posição; cada posição terá o mesmo peso na estimativa. As três seeds não serão tratadas como três posições independentes.

## 10.4. Utilidade esperada

A utilidade da escolha será 1 quando o algoritmo coincidir com a jogada de referência e 0 caso contrário.

\[
EU(a\mid x,B)=
P(Q=1\mid a,x,B)\cdot1+
P(Q=0\mid a,x,B)\cdot0
\]

Logo:

\[
\boxed{
a^*=\arg\max_{a\in\{Minimax,GPN\text{-}MCTS\}}EU(a\mid x,B)
}
\]

Em caso de igualdade, a híbrida escolhe Minimax.

A MEU representa a utilidade esperada da escolha do algoritmo segundo a concordância com a referência; não representa diretamente a probabilidade de vencer a partida.

## 10.5. Orçamentos e política fixa de comparação

A tabela de probabilidades será construída separadamente para os seis orçamentos:

- 100 ms;
- 300 ms;
- 1.000 ms;
- 500 transições;
- 2.000 transições;
- 5.000 transições.

O custo de extração de características e seleção será incluído no orçamento. O algoritmo escolhido receberá somente o tempo ou as transições restantes.

Além da híbrida, será avaliada uma **política fixa de referência**: para cada orçamento, escolher sempre o algoritmo com maior concordância média no conjunto de calibração, independentemente do contexto. Em caso de empate, escolher Minimax.

A política híbrida não consultará Stockfish durante a partida ou a decisão.

## 10.6. Checklist de Clara

- [ ] Implementar a classificação de abertura, meio-jogo e final.
- [ ] Implementar a classificação tática/não tática.
- [ ] Implementar os seis contextos.
- [ ] Registrar ramificação, diferença de material e xeque.
- [ ] Ler os resultados de calibração produzidos pelo módulo de experimentos.
- [ ] Calcular probabilidades com suavização de Laplace.
- [ ] Construir tabelas para cada orçamento.
- [ ] Implementar o cálculo da utilidade esperada.
- [ ] Selecionar o algoritmo de maior MEU.
- [ ] Implementar desempate fixo em favor do Minimax.
- [ ] Encaminhar a posição a apenas um algoritmo.
- [ ] Descontar o custo do seletor do orçamento disponível.
- [ ] Registrar contexto, probabilidades, algoritmo escolhido e custo.
- [ ] Garantir que não haja consulta ao Stockfish durante a decisão.
- [ ] Garantir separação entre calibração e teste.
- [ ] Comparar híbrida, agentes isolados e política fixa.

**Estudar:** teoria da decisão, crenças, probabilidades condicionais, resultados, função de utilidade, Máxima Utilidade Esperada, classificação de estados e separação de dados.

**Entrega:** `algorithms/hybrid.py`, módulo de características, tabela MEU, testes e documentação da política.

---

# 11. Análise de complexidade

A análise será teórica e empírica.

## 11.1. Variáveis

| Símbolo | Significado |
|---|---|
| \(b\) | Fator de ramificação médio |
| \(d\) | Profundidade em meios-lances |
| \(h\) | Comprimento do caminho de seleção no GPN-MCTS |
| \(I\) | Número de iterações do GPN-MCTS |
| \(L\) | Limite de playout, fixado em 80 meios-lances |
| \(N\) | Número de nós armazenados no GPN-MCTS |
| \(b_j\) | Número de filhos de um nó no nível \(j\) |

## 11.2. Minimax

Para fator de ramificação aproximadamente uniforme:

\[
T(d)=O(b^d)
\]

Com busca em profundidade, armazenando os sucessores necessários nos níveis ativos:

\[
S(d)=O(bd)
\]

## 11.3. Minimax com Alfa-Beta

- **Pior caso:** \(O(b^d)\).
- **Melhor caso clássico, com ordenação ideal:** \(O(b^{d/2})\).
- **Espaço:** \(O(bd)\), sem tabela de transposição.

A poda reduz a quantidade de nós examinados, mas não altera a resposta Minimax.

## 11.4. GPN-MCTS

Para uma iteração, o custo inclui seleção, playout e atualização dos números de prova. Como PNRank ordena os filhos dos nós atualizados:

\[
T_{\text{iteração}}=
O\left(
h+L+\sum_{j=1}^{h}b_j\log b_j
\right)
\]

Para \(I\) iterações:

\[
T(I)=
O\left(
I\left(h+L+\sum_{j=1}^{h}b_j\log b_j\right)
\right)
\]

Memória:

\[
S=O(N)
\]

O modelo considera os nós armazenados, filhos, visitas, recompensas e números de prova. É a análise da variante implementada pelo grupo, não um limite universal de toda implementação GPN-MCTS.

## 11.5. Busca híbrida

A seleção examina o tabuleiro, gera informações de contexto e consulta uma tabela de tamanho fixo:

\[
T_{\text{seleção}}=O(64+b)
\]

A busca completa é:

\[
T_{\text{híbrida}}=
T_{\text{seleção}}+
T_{\text{algoritmo escolhido}}
\]

A híbrida executa apenas um dos algoritmos por decisão. A memória adicional da tabela MEU é constante em relação ao tamanho da árvore.

## 11.6. Análise empírica

Serão registrados:

- fator de ramificação;
- profundidade atingida;
- nós visitados, expandidos e criados;
- transições;
- podas Alfa-Beta;
- iterações e playouts;
- atualizações de prova e ordenações PNRank;
- tempo real e CPU;
- memória;
- custo de seleção da híbrida.

Gráficos obrigatórios:

1. qualidade × orçamento;
2. tempo × transições;
3. nós visitados e expandidos por algoritmo;
4. profundidade atingida;
5. memória;
6. qualidade por contexto;
7. custo por iteração do GPN-MCTS;
8. desempenho nas partidas.

A análise empírica será usada para observar tendências, não para afirmar uma complexidade assintótica exata.

---

# 12. Organização do repositório e integração

## 12.1. Situação atual

A versão atual do repositório ainda contém a implementação antiga de Pokémon. A migração para xadrez seguirá este padrão:

- `game/`: substituir regras, estados e ações Pokémon pelas regras de xadrez;
- `algorithms/`: manter o contrato de agente e implementar Minimax, GPN-MCTS e híbrida;
- `experiments/`: unificar orçamento, contadores, medições e resultados;
- `docs/`: substituir as especificações antigas pelas definições deste documento;
- `tests/`: criar testes de regras, heurística, algoritmos e integração;
- `README.md`: atualizar instruções e descrição do projeto.

A branch `main` será a branch de integração. Código antigo será consultado quando útil, mas não será incorporado sem adaptação e testes.

## 12.2. Contrato único dos agentes

Todos os agentes implementarão:

```python
choose_action(state, budget) -> Decision
```

Orçamento único:

```python
@dataclass(frozen=True)
class Budget:
    max_time_ms: int | None = None
    max_transitions: int | None = None
```

Cada experimento utiliza um tipo de limite por vez: tempo ou transições.

Retorno único:

```python
@dataclass
class Decision:
    action: Action
    nodes_visited: int
    nodes_expanded: int
    transitions: int
    max_depth_reached: int
    extra: dict
```

`extra` armazena métricas específicas, como podas, iterações, playouts, provas e algoritmo escolhido.

Todos os contadores terão o mesmo significado em todos os módulos.

## 12.3. Estrutura final

```text
edb2-projeto3/
│
├── game/
│   ├── chess_rules.py
│   ├── evaluation.py
│   ├── features.py
│   └── positions.py
│
├── algorithms/
│   ├── base.py
│   ├── minimax.py
│   ├── gpn_mcts.py
│   ├── hybrid.py
│   └── match.py
│
├── experiments/
│   ├── metrics.py
│   ├── reference.py
│   ├── runner.py
│   ├── results_storage.py
│   ├── analysis.py
│   ├── data/
│   ├── results/
│   └── plots/
│
├── tests/
│   ├── test_chess_rules.py
│   ├── test_heuristic.py
│   ├── test_minimax.py
│   ├── test_gpn_mcts.py
│   ├── test_hybrid.py
│   └── test_reproducibility.py
│
├── docs/
│   ├── PLANO_PROJETO_XADREZ_EDBII.md
│   ├── game_specification.md
│   ├── experimental_protocol.md
│   ├── hybrid_strategy.md
│   └── literature_matrix.md
│
├── paper/
│   └── artigo.tex
│
├── README.md
└── requirements.txt
```

## 12.4. Padrão Git

- `main`: branch de integração.
- Cada integrante trabalha em branch própria criada a partir da `main` atualizada.
- Alterações entram por Pull Request.
- Outro integrante revisa cada módulo.
- Cada experimento registra o hash do commit utilizado.
- Resultados brutos não são editados manualmente.

---

# 13. Divisão de trabalho e checklists

## Integrante 1 — Matheus Bulhões

**Frente:** simulador de xadrez.

**Arquivos:** `game/chess_rules.py`, `game/positions.py` e `tests/test_chess_rules.py`.

- [ ] Instalar e fixar `python-chess`.
- [ ] Implementar o estado `chess.Board` com histórico.
- [ ] Implementar consulta de movimentos legais.
- [ ] Implementar transição de estado sem alterar o original.
- [ ] Implementar xeque, xeque-mate e afogamento.
- [ ] Testar roque, *en passant* e todas as promoções.
- [ ] Implementar empates automáticos e reivindicáveis.
- [ ] Implementar serialização e renderização.
- [ ] Publicar a interface comum.
- [ ] Criar testes automatizados para as regras.

**Estudar:** regras do xadrez, `python-chess`, objetos mutáveis, cópia de estado, FEN, PGN e testes.

**Entrega:** motor de regras funcional, interface comum e testes.

## Integrante 2 — Flavio Neves

**Frente:** Minimax, Alfa-Beta e heurística.

**Arquivos:** `algorithms/minimax.py`, `game/evaluation.py` e testes correspondentes.

- [ ] Implementar a heurística completa.
- [ ] Testar simetria e normalização.
- [ ] Implementar Minimax.
- [ ] Implementar Alfa-Beta.
- [ ] Implementar aprofundamento iterativo.
- [ ] Implementar ordenação de movimentos.
- [ ] Implementar interrupção por orçamento.
- [ ] Testar mate e posições táticas.
- [ ] Comparar com Minimax sem poda em árvores pequenas.
- [ ] Registrar métricas.
- [ ] Documentar complexidade temporal e espacial.

**Estudar:** Minimax, Alfa-Beta, MAX/MIN, fator de ramificação, profundidade, heurísticas e complexidade.

**Entrega:** Minimax funcional, heurística, testes e análise de complexidade.

## Integrante 3 — Gustavo

**Frente:** GPN-MCTS.

**Arquivos:** `algorithms/gpn_mcts.py` e testes correspondentes.

- [ ] Estudar o artigo original de GPN-MCTS.
- [ ] Estudar MCTS, UCT e Proof-Number Search.
- [ ] Implementar nós e filhos.
- [ ] Implementar provas por jogador.
- [ ] Implementar propagação AND/OR.
- [ ] Implementar PNRank e UCT-PN.
- [ ] Implementar playouts de até 80 meios-lances.
- [ ] Integrar Score-Bounded MCTS.
- [ ] Implementar retropropagação.
- [ ] Garantir que heurística não seja prova.
- [ ] Testar provas e refutações em árvores pequenas.
- [ ] Garantir ação legal e reprodutibilidade.
- [ ] Registrar métricas e documentar complexidade.

**Estudar:** MCTS, UCT, Proof-Number Search, números de prova, AND/OR, PNRank, Score-Bounded MCTS, playout e backpropagation.

**Entrega:** GPN-MCTS funcional, testes, documentação da variante e análise de complexidade.

## Integrante 4 — Clara

**Frente:** busca híbrida por Máxima Utilidade Esperada.

**Arquivos:** `algorithms/hybrid.py`, `game/features.py` e documentação da estratégia.

- [ ] Implementar classificação de fase.
- [ ] Implementar classificação tática.
- [ ] Implementar os seis contextos.
- [ ] Extrair ramificação, diferença de material e xeque.
- [ ] Receber resultados de calibração.
- [ ] Calcular probabilidades com suavização de Laplace.
- [ ] Construir tabelas por contexto e orçamento.
- [ ] Implementar a função de utilidade esperada.
- [ ] Implementar seleção do algoritmo de maior MEU.
- [ ] Implementar desempate em favor do Minimax.
- [ ] Encaminhar o estado a somente um algoritmo.
- [ ] Incluir o custo de seleção no orçamento.
- [ ] Registrar contexto, probabilidades e algoritmo escolhido.
- [ ] Garantir separação entre calibração e teste.
- [ ] Comparar a híbrida com os agentes isolados e a política fixa.

**Estudar:** teoria da decisão, crenças, probabilidades condicionais, utilidade, MEU, classificação de estados e separação de dados.

**Entrega:** busca híbrida, tabela MEU, testes e documentação.

## Integrante 5 — Victor Alves

**Frente:** experimentação, métricas e visualização.

**Arquivos:** `experiments/`, dados, resultados e gráficos.

- [ ] Obter partidas padrão públicas do Lichess.
- [ ] Extrair posições legais e preservar PGNs.
- [ ] Classificar posições nos seis contextos.
- [ ] Selecionar 20 posições por contexto.
- [ ] Separar calibração e teste sem compartilhar partidas.
- [ ] Configurar Stockfish 19 como referência externa.
- [ ] Implementar concordância top-1 e perda em centipeões.
- [ ] Unificar orçamento e contadores.
- [ ] Implementar medição de tempo e CPU.
- [ ] Implementar medição de memória em execução separada.
- [ ] Executar orçamentos de tempo.
- [ ] Executar orçamentos de transições.
- [ ] Executar três seeds do GPN-MCTS.
- [ ] Gerar os resultados de calibração.
- [ ] Entregar a tabela de calibração para Clara.
- [ ] Executar a avaliação final.
- [ ] Executar as 36 partidas completas.
- [ ] Gerar CSVs, tabelas e gráficos.
- [ ] Analisar complexidade teórica e empírica.

**Estudar:** PGN, `python-chess`, métricas, medição de desempenho, pandas, visualização, estatística descritiva e complexidade.

**Entrega:** conjunto de dados, runner, resultados, métricas, gráficos e análise experimental.

---

# 14. Ordem de integração

A integração do grupo seguirá esta sequência:

1. **Matheus:** disponibiliza o motor e a interface comum.
2. **Flavio:** implementa Minimax e a heurística compartilhada.
3. **Gustavo:** implementa GPN-MCTS usando a mesma interface e heurística.
4. **Victor:** prepara posições, referência externa e instrumentação.
5. **Clara:** implementa a seleção contextual e a MEU com os dados de calibração.
6. **Victor:** executa a avaliação final e as partidas.
7. **Todos:** interpretam os resultados, revisam o artigo e preparam o seminário.

Cada módulo deverá passar pelos testes antes de ser integrado à `main`.

---

# 15. Artigo e exigências da professora

O artigo será técnico/experimental, escrito em LaTeX, utilizando o template SBC indicado pela professora.

## 15.1. Estrutura

1. Introdução: contexto, problema, lacuna, proposta, contribuição e organização.
2. Trabalhos relacionados: comparação crítica dos estudos.
3. Proposta: motor, estado, heurística, algoritmos e política híbrida.
4. Avaliação: experimentos, resultados, discussão e comparação com a literatura.
5. Conclusão.
6. Referências.
7. Declaração de uso de IA.
8. Contribuição dos autores.

## 15.2. Requisitos formais

| Requisito | Padrão |
|---|---|
| Extensão | 5 a 7 páginas, excluindo referências e anexos |
| Referências | Mínimo de 15 referências científicas |
| Formato | LaTeX |
| Template | SBC indicado pela professora |
| Repositório | Link do GitHub no PDF |
| IA | Declaração obrigatória |
| Autoria | Contribuição individual dos integrantes |
| Seminário | Presença e participação oral de todos |

## 15.3. Revisão da literatura

A revisão deverá:

- apresentar soluções existentes;
- comparar problemas, métodos, evidências e limitações;
- identificar uma lacuna sustentada por estudos;
- explicar como o experimento do grupo aborda essa lacuna;
- utilizar pelo menos 15 referências científicas.

O grupo não afirmará que “não existem trabalhos” sem busca suficiente. A contribuição será apresentada como avaliação experimental e seleção contextual no cenário definido.

## 15.4. Uso de IA

Conforme as orientações da professora, IA poderá ser utilizada como apoio complementar, por exemplo, para revisão gramatical, organização de ideias, diagramas e auxílio em código. Não poderá gerar automaticamente o texto final do artigo ou dos slides.

A declaração de uso de IA será incluída no artigo e na apresentação, informando:

- ferramenta utilizada;
- finalidade;
- forma de utilização.

Imagens, diagramas ou esquemas produzidos por IA terão legenda, indicação da ferramenta e referência conceitual associada.

## 15.5. Contribuição dos autores

O artigo apresentará a contribuição efetiva de cada integrante, de acordo com o trabalho realizado em:

- definição do problema;
- revisão da literatura;
- implementação;
- avaliação experimental;
- análise dos resultados;
- redação e apresentação.

Todos os integrantes deverão conhecer o projeto completo e participar oralmente do seminário.

## 15.6. Datas informadas pela professora

| Data | Atividade |
|---|---|
| 17/11/2026 | Acompanhamento |
| 18/11/2026 | Entrega do artigo em PDF via SIGAA |
| 19/11/2026 | Orientações para apresentação e sorteio das datas |
| 24/11/2026 | Desenvolvimento e orientação dos slides |
| 25/11/2026 | Entrega das apresentações |
| 26/11, 01/12, 03/12 e 08/12 | Seminários de 12 minutos, seguidos de perguntas |

---

# 16. Referências técnicas centrais

1. **Kowalski, J.; Soemers, D. J. N. J.; Kosakowski, S.; Winands, M. H. M.** (2025). *Generalized Proof-Number Monte-Carlo Tree Search*. ECAI 2025. DOI: [10.3233/FAIA251406](https://doi.org/10.3233/FAIA251406).
2. **FIDE.** *Laws of Chess*. [https://handbook.fide.com/chapter/E012023](https://handbook.fide.com/chapter/E012023).
3. **python-chess.** *Core documentation*. [https://python-chess.readthedocs.io/en/latest/core.html](https://python-chess.readthedocs.io/en/latest/core.html).
4. **Lichess.** *Open Database*. [https://database.lichess.org/](https://database.lichess.org/).
5. **Stockfish.** *Stockfish chess engine*. [https://stockfishchess.org/](https://stockfishchess.org/).
6. **SBC.** *Template para artigos de eventos*. [https://www.sbc.org.br/documentosinstitucionais/#publicacoes](https://www.sbc.org.br/documentosinstitucionais/#publicacoes).

A bibliografia do artigo terá no mínimo 15 referências científicas. Além das referências centrais acima, incluirá trabalhos originais e estudos sobre Minimax, Alfa-Beta, MCTS, UCT, Proof-Number Search, Score-Bounded MCTS, teoria da decisão e avaliação experimental.

---

# 17. Checklist geral do grupo

## Motor de xadrez

- [ ] Regras implementadas e testadas.
- [ ] Histórico preservado.
- [ ] Ações especiais e empates tratados.
- [ ] Interface comum disponível.

## Algoritmos

- [ ] Heurística compartilhada implementada.
- [ ] Minimax com Alfa-Beta implementado.
- [ ] GPN-MCTS implementado.
- [ ] Busca híbrida por MEU implementada.
- [ ] Todos retornam ações legais.
- [ ] Todos respeitam o mesmo contrato de orçamento e métricas.

## Experimentos

- [ ] 120 posições selecionadas.
- [ ] 60 posições de calibração.
- [ ] 60 posições de teste.
- [ ] Stockfish configurado como referência externa.
- [ ] Orçamentos de tempo executados.
- [ ] Orçamentos de transições executados.
- [ ] Memória medida separadamente.
- [ ] 36 partidas completas executadas.
- [ ] Resultados brutos preservados.
- [ ] Complexidade teórica e empírica analisada.
- [ ] Gráficos e tabelas gerados.

## Artigo e seminário

- [ ] Pelo menos 15 referências científicas.
- [ ] Lacuna sustentada pela literatura.
- [ ] Artigo no template SBC.
- [ ] Extensão de 5 a 7 páginas, sem referências e anexos.
- [ ] Resultados e limitações discutidos.
- [ ] Link do GitHub incluído.
- [ ] Declaração de uso de IA incluída.
- [ ] Contribuição individual dos autores incluída.
- [ ] Todos os integrantes preparados para apresentar e responder perguntas.

---

# Padrão final

Este documento é a especificação única do projeto. As regras, estados, ações, utilidade, heurística, parâmetros, orçamentos, métricas, algoritmos e responsabilidades aqui definidos são o padrão de implementação e avaliação do grupo.

Os três agentes serão comparados com as mesmas posições, regras e orçamento. O tempo real será a medida principal de custo computacional; transições, nós, profundidade, memória e métricas específicas complementarão a análise.

Os parâmetros permanecerão fixos durante a avaliação. Os resultados serão apresentados como forem obtidos, inclusive quando um algoritmo não superar os demais.
