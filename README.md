# edb2-projeto3

Projeto de EDB II (UFRN): comparação entre Minimax com poda Alfa-Beta, GPN-MCTS e uma busca híbrida contextual por Máxima Utilidade Esperada em partidas de xadrez.
Especificação oficial: [docs/PLANO_PROJETO_XADREZ_EDBII.md](docs/PLANO_PROJETO_XADREZ_EDBII.md).

## Como rodar
Padrão do grupo: Python 3.12. Dependências em `requirements.txt` (`python-chess`, `pytest`). Rode os comandos a partir da raiz do repositório.

```bash
# (opcional) ambiente virtual
python -m venv .venv
.venv\Scripts\activate              # Windows  (Linux/macOS: source .venv/bin/activate)

pip install -r requirements.txt
python -m pytest -q
```

## Estrutura
```text
game/            motor de xadrez e interface comum (Integrante 1)
  chess_rules.py   regras, estado (chess.Board) e interface comum — em implementação
  positions.py     posições de teste — em implementação
algorithms/      agentes
  base.py          contrato: Agent, Budget, Decision + RandomAgent, GreedyAgent
  match.py         play_match (partida entre dois agentes) + demonstração
  minimax.py       Minimax + Alfa-Beta (Integrante 2)
  mcts.py          MCTS (Integrante 3)
  gpn_mcts.py      GPN-MCTS (Integrante 3)
  hybrid.py        busca híbrida (Integrante 4)
tests/
  test_chess_rules.py  regras do xadrez — em implementação
docs/            PLANO_PROJETO_XADREZ_EDBII.md, game_specification.md, experimental_protocol.md, hybrid_strategy.md
paper/           artigo
conftest.py      permite rodar o pytest de qualquer pasta
requirements.txt dependências (python-chess, pytest)
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

## Status
- [ ] Motor de xadrez e interface comum (Integrante 1)
- [ ] Minimax + Alfa-Beta (Integrante 2)
- [ ] MCTS / GPN-MCTS (Integrante 3)
- [ ] Busca híbrida (Integrante 4)
- [ ] Benchmarks e visualização (Integrante 5)
