# edb2-projeto3

Projeto de EDB II (UFRN): comparação entre Minimax com poda Alfa-Beta, GPN-MCTS e uma busca híbrida contextual em batalhas Pokémon simplificadas.

## Como rodar
```bash
pip install -r requirements.txt
python -m pytest -q                 # testes das regras + gabarito dos cenários
python -m algorithms.match          # partida de demonstração (guloso x aleatório)
python -m tests.oracle              # recalcula o gabarito dos cenários
python -m game.pokeapi fetch        # (só para atualizar) baixa os dados da PokéAPI
```
Requer Python 3.10 ou superior. Rode os comandos a partir da raiz do repositório.

## Estrutura
```text
game/          regras, estado, interface comum e cenários (Integrante 1)
  pokemon.py     Move, PokemonSpec (dados fixos)
  state.py       BattleState, PokemonState (imutáveis)
  battle.py      legal_actions, next_state, utility, evaluate, ...
  scenarios.py   cenários sanity / trap / large / real
  pokeapi.py     conversão de dados reais da PokéAPI (cache em data/pokeapi.json)
algorithms/    agentes
  base.py        contrato: Agent, Budget, Decision + RandomAgent, GreedyAgent
  match.py       play_match (partida entre dois agentes)
  minimax.py     Minimax + Alfa-Beta (Integrante 2)
  mcts.py        MCTS (Integrante 3)
  gpn_mcts.py    GPN-MCTS (Integrante 3)
  hybrid.py      busca híbrida (Integrante 4)
tests/         testes + solver exato (oracle.py) usado como gabarito
docs/          game_specification.md, experimental_protocol.md, hybrid_strategy.md
paper/         artigo
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
Valide o agente com `tests/test_scenarios.py::EXPECTED`: nos cenários `trap_*` ele deve escolher a jogada ótima.

## Status
- [x] Regras, estado e interface (protótipo, aguardando aprovação do grupo — ver `docs/game_specification.md`)
- [x] Cenários com gabarito, solver exato, agentes Random/Greedy, `play_match`
- [ ] Minimax + Alfa-Beta (Integrante 2)
- [ ] MCTS / GPN-MCTS (Integrante 3)
- [ ] Busca híbrida (Integrante 4)
- [ ] Benchmarks e visualização (Integrante 5)
