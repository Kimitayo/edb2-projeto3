# edb2-projeto3

Projeto de EDB II (UFRN): comparação entre Minimax com poda Alfa-Beta, GPN-MCTS e uma busca híbrida contextual em batalhas Pokémon simplificadas.

## Como rodar
Requer Python 3.10 ou superior (testado com 3.14). A única dependência externa é o `pytest`; o resto é biblioteca padrão. Rode os comandos a partir da raiz do repositório.

```bash
# (opcional) ambiente virtual
python -m venv .venv
.venv\Scripts\activate              # Windows  (Linux/macOS: source .venv/bin/activate)

pip install -r requirements.txt
python -m pytest -q                 # regras, conversão da PokéAPI e gabarito dos cenários
python -m algorithms.match          # partida de demonstração (guloso x aleatório)
python -m algorithms.match trap_3v3 # idem, em outro cenário (ver game/scenarios.py)
python -m tests.oracle              # recalcula o gabarito dos cenários
```

Os dados da PokéAPI já estão em `game/data/pokeapi.json`; testes e demonstração não usam a rede.
Só para incluir Pokémon/ataques novos: edite `WANTED` em `game/pokeapi.py` e rode `python -m game.pokeapi fetch` (precisa de internet e **sobrescreve** o JSON).

## Estrutura
```text
game/            regras, estado, interface comum e cenários (Integrante 1)
  pokemon.py       Move, PokemonSpec (dados fixos)
  state.py         BattleState, PokemonState (imutáveis)
  battle.py        legal_actions, next_state, utility, evaluate, ...
  scenarios.py     cenários sanity / trap / large / real
  pokeapi.py       conversão de dados reais da PokéAPI
  data/pokeapi.json  cache local dos dados da PokéAPI
algorithms/      agentes
  base.py          contrato: Agent, Budget, Decision + RandomAgent, GreedyAgent
  match.py         play_match (partida entre dois agentes) + demonstração
  minimax.py       Minimax + Alfa-Beta (Integrante 2)
  mcts.py          MCTS (Integrante 3)
  gpn_mcts.py      GPN-MCTS (Integrante 3)
  hybrid.py        busca híbrida (Integrante 4)
tests/
  test_rules.py      regras da batalha
  test_scenarios.py  gabarito dos cenários (EXPECTED)
  test_pokeapi.py    conversão dos dados da PokéAPI
  oracle.py          solver exato usado só como gabarito
docs/            game_specification.md, experimental_protocol.md, hybrid_strategy.md
paper/           artigo
conftest.py      permite rodar o pytest de qualquer pasta
requirements.txt dependências (pytest)
```

## Exemplo mínimo para quem implementa um algoritmo
```python
from game import legal_actions, next_state, is_terminal, utility, evaluate
from algorithms.base import Agent, Budget, Decision

class MeuAgente(Agent):
    name = "meu"
    def choose_action(self, state, budget: Budget) -> Decision:
        ...
        return Decision(action=melhor, nodes_expanded=n)
```
Valide o agente com `tests/test_scenarios.py::EXPECTED`: nos cenários `trap_*` e `real_*` ele deve escolher a jogada ótima.

## Status
- [x] Regras, estado e interface (protótipo, aguardando aprovação do grupo — ver `docs/game_specification.md`)
- [x] Cenários com gabarito, solver exato, agentes Random/Greedy, `play_match`
- [x] Dados reais da PokéAPI (cenários `real_*`) — aguardando aprovação do grupo
- [ ] Minimax + Alfa-Beta (Integrante 2)
- [ ] MCTS / GPN-MCTS (Integrante 3)
- [ ] Busca híbrida (Integrante 4)
- [ ] Benchmarks e visualização (Integrante 5)
