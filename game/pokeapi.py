"""Dados reais da PokéAPI (https://pokeapi.co) convertidos para Move/PokemonSpec.

Duas partes:
- fetch (ÚNICA parte que usa rede): `python -m game.pokeapi fetch` baixa os
  tipos e os Pokémon/ataques listados em WANTED e salva só os campos usados em
  game/data/pokeapi.json (versionado). Segue a política de uso justo da API
  (cache local) e deixa testes/experimentos offline e reprodutíveis.
- builder (offline): lê o JSON e monta os specs de um confronto.

Simplificações (ver docs/game_specification.md, D2b):
- nível 50, IV 31, EV 0, natureza neutra;
- dano pela fórmula oficial SEM fator aleatório, precisão e crítico;
- STAB 1,5 e efetividade de tipo embutidos no dano (por isso o spec depende
  do oponente e é montado por confronto);
- dreno (`meta.drain`) vira cura fixa calculada sobre o dano daquele confronto;
- PP real dividido por 5 (mínimo 1) para manter a árvore tratável;
- sem status, alterações de atributo, habilidades e itens.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from .pokemon import Move, PokemonSpec

DATA_FILE = Path(__file__).with_name("data") / "pokeapi.json"
API = "https://pokeapi.co/api/v2"
LEVEL = 50
PP_DIVISOR = 5

TYPES = ("normal", "fire", "water", "electric", "grass", "ice", "fighting",
         "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
         "dragon", "dark", "steel", "fairy")

# O que o `fetch` baixa. Para usar um Pokémon/ataque novo, adicione aqui e
# rode `python -m game.pokeapi fetch` de novo.
WANTED = {
    "pokemon": ["breloom", "slowbro", "chansey", "gengar", "lucario",
                "snorlax", "venusaur", "machamp"],
    "move": ["tackle", "body-slam", "mach-punch", "drain-punch", "seed-bomb",
             "giga-drain", "water-gun", "scald", "slack-off", "soft-boiled",
             "recover", "shadow-ball", "aura-sphere", "close-combat",
             "psychic", "pound"],
}


# --- fetch (rede) -----------------------------------------------------------
def _get(path: str) -> dict:
    from urllib.request import Request, urlopen
    req = Request(f"{API}/{path}", headers={"User-Agent": "edb2-projeto3"})
    with urlopen(req, timeout=30) as r:
        return json.load(r)


def _trim_pokemon(d: dict) -> dict:
    return {"stats": {s["stat"]["name"]: s["base_stat"] for s in d["stats"]},
            "types": [t["type"]["name"] for t in sorted(d["types"], key=lambda t: t["slot"])]}


def _trim_move(d: dict) -> dict:
    meta = d.get("meta") or {}
    return {"power": d["power"], "pp": d["pp"], "type": d["type"]["name"],
            "damage_class": d["damage_class"]["name"],
            "healing": meta.get("healing", 0) or 0, "drain": meta.get("drain", 0) or 0}


def _trim_type(d: dict) -> dict:
    rel = d["damage_relations"]
    return {k: sorted(t["name"] for t in rel[k])
            for k in ("double_damage_to", "half_damage_to", "no_damage_to")}


def fetch(path: Path = DATA_FILE) -> dict:
    data = {"type": {t: _trim_type(_get(f"type/{t}")) for t in TYPES},
            "pokemon": {p: _trim_pokemon(_get(f"pokemon/{p}")) for p in WANTED["pokemon"]},
            "move": {m: _trim_move(_get(f"move/{m}")) for m in WANTED["move"]}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return data


# --- builder (offline) ------------------------------------------------------
@lru_cache(maxsize=None)
def load() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def stats(pokemon: str) -> dict[str, int]:
    """Atributos no nível 50 (IV 31, EV 0, natureza neutra)."""
    base = load()["pokemon"][pokemon]["stats"]
    out = {}
    for name, b in base.items():
        core = (2 * b + 31) * LEVEL // 100
        out[name] = core + LEVEL + 10 if name == "hp" else core + 5
    return out


def types_of(pokemon: str) -> list[str]:
    return load()["pokemon"][pokemon]["types"]


def effectiveness(move_type: str, defender_types: list[str]) -> float:
    rel = load()["type"][move_type]
    mult = 1.0
    for t in defender_types:
        if t in rel["no_damage_to"]:
            mult *= 0.0
        elif t in rel["double_damage_to"]:
            mult *= 2.0
        elif t in rel["half_damage_to"]:
            mult *= 0.5
    return mult


def damage(attacker: str, defender: str, move: str) -> int:
    """Dano determinístico (sem fator aleatório, precisão e crítico)."""
    mv = load()["move"][move]
    if not mv["power"]:
        return 0
    a, d = stats(attacker), stats(defender)
    if mv["damage_class"] == "physical":
        atk, dfn = a["attack"], d["defense"]
    else:
        atk, dfn = a["special-attack"], d["special-defense"]
    base = (2 * LEVEL // 5 + 2) * mv["power"] * atk // dfn // 50 + 2
    stab = 1.5 if mv["type"] in types_of(attacker) else 1.0
    return int(base * stab * effectiveness(mv["type"], types_of(defender)))


def make_move(attacker: str, defender: str, move: str, unlimited: bool = False) -> Move:
    mv = load()["move"][move]
    if mv["drain"] < 0:
        raise ValueError(f"{move}: ataques com recuo não são suportados")
    if not mv["power"] and not mv["healing"]:
        raise ValueError(f"{move}: ataque de status sem cura não é suportado")
    dmg = damage(attacker, defender, move)
    heal = mv["healing"] * stats(attacker)["hp"] // 100 + mv["drain"] * dmg // 100
    pp = None if unlimited else max(1, mv["pp"] // PP_DIVISOR)
    return Move(move, damage=dmg, heal=heal, max_pp=pp)


def make_spec(pokemon: str, moves: list[str], opponent: str, unlimited: str) -> PokemonSpec:
    """Spec de `pokemon` contra `opponent`; `unlimited` é o ataque de PP ilimitado."""
    if unlimited not in moves:
        raise ValueError(f"{unlimited} não está entre os ataques de {pokemon}")
    return PokemonSpec(pokemon, stats(pokemon)["hp"],
                       tuple(make_move(pokemon, opponent, m, m == unlimited) for m in moves))


def build_matchup(p0: str, moves0: list[str], unlimited0: str,
                  p1: str, moves1: list[str], unlimited1: str) -> tuple[PokemonSpec, PokemonSpec]:
    return (make_spec(p0, moves0, p1, unlimited0), make_spec(p1, moves1, p0, unlimited1))


def first_player(p0: str, p1: str) -> int:
    """Quem tem mais velocidade começa (empate: jogador 0)."""
    return 1 if stats(p1)["speed"] > stats(p0)["speed"] else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        d = fetch()
        print(f"{DATA_FILE}: {len(d['type'])} tipos, {len(d['pokemon'])} Pokémon, "
              f"{len(d['move'])} ataques")
    else:
        print("uso: python -m game.pokeapi fetch")
