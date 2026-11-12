"""Testes das regras da batalha."""
import pytest

from game import (Move, PokemonSpec, initial_state, legal_actions, next_state,
                  is_terminal, get_winner, utility, evaluate, state_features,
                  reset_transition_count, transition_count, to_dict, render)

HIT = Move("Hit", damage=30)
NUKE = Move("Nuke", damage=60, max_pp=1)
HEAL = Move("Heal", heal=50, max_pp=1)
A = PokemonSpec("A", 100, (HIT, NUKE, HEAL))
B = PokemonSpec("B", 50, (HIT,))


def start(first=0, max_plies=20):
    return initial_state(A, B, first, max_plies)


def test_initial_state():
    s = start()
    assert s.team[0].hp == 100 and s.team[1].hp == 50
    assert s.to_move == 0 and s.ply == 0
    assert s.team[0].pp == (None, 1, 1)
    assert legal_actions(s) == (0, 1, 2)


def test_damage_switches_turn_and_is_immutable():
    s0 = start()
    s1 = next_state(s0, 0)
    assert s1.team[1].hp == 20 and s1.to_move == 1 and s1.ply == 1
    assert s0.team[1].hp == 50 and s0.to_move == 0  # original não mudou


def test_pp_is_consumed_and_move_becomes_illegal():
    s = next_state(start(max_plies=20), 2)       # A cura (gasta PP)
    s = next_state(s, 0)                          # B ataca
    assert 2 not in legal_actions(s)
    with pytest.raises(ValueError):
        next_state(s, 2)


def test_heal_capped_at_max_hp():
    s = next_state(start(), 2)
    assert s.team[0].hp == 100


def test_knockout_and_winner():
    s = next_state(start(), 1)                    # 60 dano em 50 HP
    assert s.team[1].hp == 0 and is_terminal(s)
    assert get_winner(s) == 0
    assert utility(s, 0) == 1.0 and utility(s, 1) == -1.0
    assert legal_actions(s) == ()


def test_draw_by_ply_limit():
    s = start(max_plies=1)
    s = next_state(s, 2)                          # cura, ninguém desmaia
    assert is_terminal(s) and get_winner(s) is None
    assert utility(s, 0) == 0.0


def test_utility_requires_terminal():
    with pytest.raises(ValueError):
        utility(start(), 0)


def test_evaluate_range_and_symmetry():
    s = next_state(start(), 0)
    assert -1 <= evaluate(s, 0) <= 1
    assert evaluate(s, 0) == pytest.approx(-evaluate(s, 1))


def test_states_are_hashable_and_equal_by_value():
    assert start() == start() and hash(start()) == hash(start())
    assert len({start(), start()}) == 1


def test_transition_counter():
    reset_transition_count()
    next_state(next_state(start(), 0), 0)
    assert transition_count() == 2


def test_spec_validation():
    with pytest.raises(ValueError):
        PokemonSpec("X", 10, (NUKE,))             # sem ataque de PP ilimitado
    with pytest.raises(ValueError):
        Move("Bad", damage=-1)


def test_helpers():
    s = start()
    assert to_dict(s)["team"][0]["name"] == "A"
    assert "J0" in render(s)
    assert state_features(s)["branching"] == 3
