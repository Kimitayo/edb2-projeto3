# Especificação do jogo e interface comum

Status: **PROPOSTA (Integrante 1) — aguardando aprovação do grupo.**
Qualquer mudança neste documento precisa ser combinada com todos, porque afeta os 4 algoritmos e os experimentos.
Protocolo de experimentos (orçamento, cenários, gabarito): ver [experimental_protocol.md](experimental_protocol.md).

## D1. Modelo de jogo
- 2 jogadores (0 e 1), soma zero, determinístico, informação completa.
- **Turnos alternados** (um jogador age por vez). Simplificação em relação ao Pokémon oficial (escolhas simultâneas). Quem começa é definido pelo cenário.
- Batalha 1v1, de 1 a 4 ataques por Pokémon.

## D2. Ataques (`game/pokemon.py`)
- Campos: `damage` (dano fixo), `heal` (cura própria), `max_pp` (usos; `None` = ilimitado).
- Todo Pokémon tem ao menos 1 ataque com PP ilimitado (garante que sempre há ação legal).
- Ordem de resolução: gasta PP → dano no oponente (HP mín. 0) → cura própria (HP máx. = max_hp) → passa a vez.

**Por que PP e cura já no MVP:** sem eles, a jogada ótima é sempre "o ataque mais forte" (verificado com solver exato, cenário `sanity_mvp`) e nenhum algoritmo se diferenciaria.

## D3. Fim de jogo
- HP de um Pokémon = 0 → o outro vence.
- `ply == max_plies` → **empate** (garante término mesmo com cura).

## D4. Estado (`game/state.py`)
- `BattleState` imutável e hashable: `team` (HP e PP de cada lado), `to_move`, `ply`, `max_plies`.
- Dados fixos (espécie, ataques, HP máximo) ficam em `PokemonSpec`.
- `state.key()` dá uma tupla compacta para tabelas de transposição.

## D5. Interface de regras (`game/battle.py`)
| Função | Retorno |
|---|---|
| `legal_actions(s)` | tupla de índices de ataques (vazia se terminal) |
| `next_state(s, a)` | novo estado (`ValueError` se ilegal) |
| `is_terminal(s)` | bool |
| `get_winner(s)` | 0, 1 ou `None` (empate / em andamento) |
| `utility(s, p)` | +1 / 0 / −1, **só em terminal** (resultado exato — base dos proof numbers) |
| `evaluate(s, p)` | heurística em [−1, 1] (diferença de frações de HP) |
| `state_features(s)` | dict de atributos para a busca híbrida |
| `to_dict(s)` / `render(s)` | serialização / texto para visualização |
| `reset_transition_count()` / `transition_count()` | contador de chamadas a `next_state` (métrica de custo comum) |

## D6. Interface dos agentes (`algorithms/base.py`)
```python
class Agent:
    name: str
    def choose_action(self, state, budget: Budget) -> Decision
```
- `Budget(max_transitions, max_time_s, max_depth)`.
- `Decision(action, nodes_expanded, nodes_visited, max_depth_reached, extra)`.
- Baselines prontos: `RandomAgent(seed)` e `GreedyAgent` (1 jogada + `evaluate`).
- Cada algoritmo fica no seu arquivo: `minimax.py`, `mcts.py`, `gpn_mcts.py`, `hybrid.py`.
- Partidas entre agentes: `algorithms/match.py::play_match`.

## Pontos em aberto
1. Heurística `evaluate` única ou por algoritmo?
2. Expansões futuras (tipos, atributos) — só se não comprometer o prazo.
