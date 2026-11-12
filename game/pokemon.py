"""Definições estáticas: ataques (Move) e espécies (PokemonSpec).

Esses objetos NÃO mudam durante a batalha. Tudo que muda (HP, PP)
fica em game/state.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Move:
    """Um ataque.

    damage : dano fixo causado ao oponente (determinístico).
    heal   : HP recuperado pelo próprio usuário (limitado ao HP máximo).
    max_pp : número de usos; None = ilimitado.
    """
    name: str
    damage: int = 0
    heal: int = 0
    max_pp: Optional[int] = None

    def __post_init__(self) -> None:
        if self.damage < 0 or self.heal < 0:
            raise ValueError(f"{self.name}: damage/heal não podem ser negativos")
        if self.max_pp is not None and self.max_pp < 1:
            raise ValueError(f"{self.name}: max_pp deve ser >= 1 ou None")


@dataclass(frozen=True)
class PokemonSpec:
    """Espécie/configuração de um Pokémon (dados fixos)."""
    name: str
    max_hp: int
    moves: tuple[Move, ...]

    def __post_init__(self) -> None:
        if self.max_hp < 1:
            raise ValueError(f"{self.name}: max_hp deve ser >= 1")
        if not 1 <= len(self.moves) <= 4:
            raise ValueError(f"{self.name}: deve ter de 1 a 4 ataques")
        # Regra que garante que todo estado não terminal tem ao menos 1 ação.
        if not any(m.max_pp is None for m in self.moves):
            raise ValueError(f"{self.name}: precisa de ao menos 1 ataque com PP ilimitado")
