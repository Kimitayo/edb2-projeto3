"""Representação do estado da batalha.

Decisões:
- Estados são IMUTÁVEIS (frozen dataclass): next_state sempre devolve um novo
  objeto. Isso evita bugs de "desfazer jogada" nas buscas e permite usar o
  estado como chave de dicionário (tabela de transposição, cache de nós).
- Jogadores são 0 e 1. O índice do jogador é o índice em `team`.
- `ply` conta jogadas individuais (meio-turnos). Ao atingir `max_plies`,
  a batalha termina em empate. Isso garante término mesmo com ataques de cura.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .pokemon import PokemonSpec

PLAYERS = (0, 1)


def opponent(player: int) -> int:
    return 1 - player


@dataclass(frozen=True)
class PokemonState:
    spec: PokemonSpec
    hp: int
    pp: tuple[Optional[int], ...]  # PP restante por ataque (None = ilimitado)

    @staticmethod
    def fresh(spec: PokemonSpec) -> "PokemonState":
        return PokemonState(spec, spec.max_hp, tuple(m.max_pp for m in spec.moves))

    @property
    def fainted(self) -> bool:
        return self.hp <= 0

    def key(self) -> tuple:
        return (self.hp, self.pp)


@dataclass(frozen=True)
class BattleState:
    team: tuple[PokemonState, PokemonState]  # team[p] = Pokémon do jogador p
    to_move: int                              # jogador da vez (0 ou 1)
    ply: int = 0                              # jogadas já realizadas
    max_plies: int = 40                       # limite -> empate

    def key(self) -> tuple:
        """Chave compacta e hashable (sem os dados estáticos das espécies).
        Use para tabelas de transposição quando as espécies forem fixas."""
        return (self.team[0].key(), self.team[1].key(), self.to_move, self.ply)

    def with_team(self, p: int, mon: PokemonState) -> "BattleState":
        t = list(self.team)
        t[p] = mon
        return replace(self, team=(t[0], t[1]))


def initial_state(spec0: PokemonSpec, spec1: PokemonSpec,
                  first_player: int = 0, max_plies: int = 40) -> BattleState:
    if first_player not in PLAYERS:
        raise ValueError("first_player deve ser 0 ou 1")
    return BattleState((PokemonState.fresh(spec0), PokemonState.fresh(spec1)),
                       first_player, 0, max_plies)
