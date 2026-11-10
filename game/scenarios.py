"""Cenários de batalha pré-definidos.

O gabarito de cada cenário (valor com jogo perfeito e jogadas ótimas na raiz)
foi calculado pelo solver exato de tests/oracle.py e está fixado em
tests/test_scenarios.py. Ao criar um cenário novo, rode:

    python -m tests.oracle

para obter o gabarito e só então adicioná-lo aos testes.

Categorias:
- sanity : decisão trivial (verifica se o algoritmo não é "burro")
- trap   : jogada ótima ÚNICA e o agente guloso (1 jogada) ERRA
           -> mede qualidade de decisão
- large  : árvore grande, sem gabarito obrigatório -> mede custo/orçamento
"""
from __future__ import annotations

from .pokemon import Move, PokemonSpec
from .state import initial_state

# --- Ataques ----------------------------------------------------------------
TACKLE = Move("Tackle", damage=20)
SCRATCH = Move("Scratch", damage=15)
EMBER = Move("Ember", damage=25)
POUND = Move("Pound", damage=10)
BITE = Move("Bite", damage=40)
HYPER_BEAM = Move("Hyper Beam", damage=45, max_pp=1)
RECOVER = Move("Recover", heal=40, max_pp=2)
GIGA_DRAIN = Move("Giga Drain", damage=15, heal=15, max_pp=3)

# --- Espécies ---------------------------------------------------------------
CHARMANDER = PokemonSpec("Charmander", 100, (SCRATCH, EMBER))
EEVEE = PokemonSpec("Eevee", 100, (TACKLE, HYPER_BEAM, RECOVER, GIGA_DRAIN))

# Armadilha 1: guardar a cura forte (Drain Punch) para depois
BRELOOM = PokemonSpec("Breloom", 100, (POUND, Move("Drain Punch", 20, 40, 2)))
SCYTHER = PokemonSpec("Scyther", 60, (TACKLE, Move("X-Scissor", 50, 0, 3)))

# Armadilha 2: não gastar o golpe único cedo
MANKEY = PokemonSpec("Mankey", 60, (BITE, Move("Low Kick", 10, 0, 1),
                                    Move("Close Combat", 50, 15, 1)))
SLOWBRO = PokemonSpec("Slowbro", 120, (Move("Water Gun", 30), Move("Slack Off", 0, 25, 3)))

# Armadilha 3: 2 ataques x 3 ataques, árvore média
MACHOKE = PokemonSpec("Machoke", 100, (TACKLE, Move("Dynamic Punch", 50, 40, 2)))
VENUSAUR = PokemonSpec("Venusaur", 120, (POUND, Move("Petal Dance", 30, 25, 3),
                                         Move("Leech Seed", 25, 15, 1)))

# Armadilha 4: 3 x 3 ataques, árvore maior
LUCARIO = PokemonSpec("Lucario", 120, (Move("Force Palm", 15, 15),
                                       Move("Aura Sphere", 40, 0, 1),
                                       Move("Drain Kick", 40, 15, 3)))
GOLDUCK = PokemonSpec("Golduck", 100, (POUND, Move("Hydro Drain", 40, 25, 2),
                                       Move("Aqua Heal", 20, 25, 3)))


SCENARIOS = {
    # sanity: sem PP/cura -> sempre o ataque mais forte (mostra por que o MVP puro é trivial)
    "sanity_mvp": lambda: initial_state(CHARMANDER, CHARMANDER, 0, 40),
    # trap
    "trap_save_heal": lambda: initial_state(BRELOOM, SCYTHER, 0, 16),
    "trap_save_nuke": lambda: initial_state(MANKEY, SLOWBRO, 0, 16),
    "trap_2v3": lambda: initial_state(MACHOKE, VENUSAUR, 0, 16),
    "trap_3v3": lambda: initial_state(LUCARIO, GOLDUCK, 0, 16),
    # large
    "large_4v4": lambda: initial_state(EEVEE, EEVEE, 0, 30),
}

CATEGORY = {name: name.split("_")[0] for name in SCENARIOS}

