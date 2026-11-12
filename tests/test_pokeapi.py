"""Testes da conversão PokéAPI -> Move/PokemonSpec (offline, usa game/data/pokeapi.json).

Valores esperados calculados à mão a partir do JSON (nível 50, IV 31, EV 0).
"""
import pytest

from game import pokeapi
from game.pokeapi import (build_matchup, damage, effectiveness, first_player,
                          make_move, make_spec, stats)


def test_stats_level_50():
    s = stats("lucario")  # base: hp 70, atk 110, def 70, spa 115, spd 70
    assert (s["hp"], s["attack"], s["defense"], s["special-attack"]) == (145, 130, 90, 135)
    assert stats("venusaur")["hp"] == 155


def test_effectiveness():
    assert effectiveness("fighting", ["ghost", "poison"]) == 0.0
    assert effectiveness("fighting", ["normal"]) == 2.0
    assert effectiveness("fighting", ["grass", "poison"]) == 0.5
    assert effectiveness("water", ["fire", "rock"]) == 4.0


def test_damage_special_with_stab_and_resistance():
    # floor(floor(22*80*135/120)/50)+2 = 41; *1.5 (STAB) *0.5 (poison) = 30
    assert damage("lucario", "venusaur", "aura-sphere") == 30


def test_damage_physical_and_immunity():
    # floor(floor(22*75*130/103)/50)+2 = 43; *1.5 *0.5 = 32
    assert damage("lucario", "venusaur", "drain-punch") == 32
    assert damage("lucario", "gengar", "drain-punch") == 0


def test_drain_becomes_fixed_heal_and_pp_is_scaled():
    mv = make_move("lucario", "venusaur", "drain-punch")
    assert (mv.damage, mv.heal, mv.max_pp) == (32, 16, 2)   # pp 10 // 5
    assert make_move("lucario", "venusaur", "drain-punch", unlimited=True).max_pp is None


def test_healing_move_uses_max_hp():
    mv = make_move("slowbro", "breloom", "slack-off")        # 50% de 170
    assert (mv.damage, mv.heal, mv.max_pp) == (0, 85, 1)


def test_unsupported_moves(monkeypatch):
    monkeypatch.setitem(pokeapi.load()["move"], "fake-recoil",
                        {"power": 120, "pp": 15, "type": "normal",
                         "damage_class": "physical", "healing": 0, "drain": -33})
    monkeypatch.setitem(pokeapi.load()["move"], "fake-status",
                        {"power": None, "pp": 20, "type": "normal",
                         "damage_class": "status", "healing": 0, "drain": 0})
    with pytest.raises(ValueError):
        make_move("snorlax", "chansey", "fake-recoil")
    with pytest.raises(ValueError):
        make_move("snorlax", "chansey", "fake-status")


def test_spec_requires_unlimited_move_in_list():
    with pytest.raises(ValueError):
        make_spec("lucario", ["aura-sphere"], "venusaur", unlimited="tackle")


def test_build_matchup_and_first_player():
    s0, s1 = build_matchup("lucario", ["aura-sphere", "drain-punch"], "aura-sphere",
                           "venusaur", ["tackle", "giga-drain"], "tackle")
    assert (s0.name, s0.max_hp, s1.name, s1.max_hp) == ("lucario", 145, "venusaur", 155)
    assert s0.moves[0].max_pp is None and s1.moves[0].max_pp is None
    assert first_player("lucario", "venusaur") == 0     # speed 90 x 80
    assert first_player("slowbro", "gengar") == 1       # speed 30 x 110
