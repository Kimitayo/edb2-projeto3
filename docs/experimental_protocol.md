# Protocolo experimental

Status: **PROPOSTA (Integrante 1) — aguardando aprovação do grupo.** O detalhamento dos experimentos é do Integrante 5.
Regras do jogo e interfaces: ver [game_specification.md](game_specification.md).

## Orçamento e métricas
- **Orçamento principal = número de chamadas a `next_state`** (`Budget.max_transitions`): é determinístico e não depende da máquina.
- O simulador conta essas chamadas (`reset_transition_count()` / `transition_count()`), então a medição é a mesma para todos os algoritmos, sem depender da contagem interna de cada um.
- `Budget.max_time_s` é opcional, só para experimentos de tempo real.
- Por decisão, `algorithms/match.py::play_match` registra: transições usadas, tempo, `nodes_expanded` e o dicionário `extra` do agente.

## Cenários e gabarito
- `game/scenarios.py`, com três categorias:
  - `sanity`: decisão trivial (verifica se o algoritmo não é "burro");
  - `trap`: jogada ótima **única** em que o agente guloso **erra** (mede a qualidade da decisão);
  - `large`: árvore grande, sem gabarito obrigatório (mede custo/orçamento).
- O gabarito é calculado pelo solver exato (`tests/oracle.py`, negamax sem poda, usado **só** como gabarito) e fixado em `tests/test_scenarios.py::EXPECTED`.
- Cenário novo: rodar `python -m tests.oracle`, conferir o resultado e só então registrar em `EXPECTED`.
- Separar os cenários usados para **definir** a política híbrida dos usados para **avaliá-la** (a definir pelos Integrantes 4 e 5).

## Pontos em aberto
1. Valores de `max_transitions` para os experimentos.
2. Quais cenários entram em cada conjunto (definição × avaliação da política híbrida).
