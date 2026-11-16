"""Demonstração visual do motor: partida com ações aleatórias no terminal.

    python -m game                         # anima a partida (seed 2026)
    python -m game --seed 7 --delay 0      # só o tabuleiro final
    python -m game --svg final.svg --pgn partida.pgn
    python -m game --ascii                 # letras em vez de símbolos Unicode

Ctrl+C interrompe. Serve como teste visual de legal_actions, next_state,
is_terminal e get_outcome funcionando juntos; não é um agente do projeto.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from .chess_rules import (ChessState, Outcome, get_outcome, initial_state,
                          is_terminal, legal_actions, next_state, render,
                          render_svg, reset_transition_count, to_pgn,
                          transition_count)

_CLEAR = "\x1b[2J\x1b[H"   # limpa a tela (sequência ANSI)


def play_random_game(seed: int = 2026, fen: Optional[str] = None,
                     max_plies: Optional[int] = None,
                     on_step=None) -> tuple[ChessState, Outcome]:
    """Joga ações legais aleatórias até o fim (ou até `max_plies`).
    `on_step(state)` é chamado após cada ação."""
    rng = random.Random(seed)
    state = initial_state(fen)
    plies = 0
    while not is_terminal(state) and (max_plies is None or plies < max_plies):
        state = next_state(state, rng.choice(legal_actions(state)))
        plies += 1
        if on_step is not None:
            on_step(state)
    return state, get_outcome(state)


def main(argv: Optional[list[str]] = None) -> Outcome:
    p = argparse.ArgumentParser(prog="python -m game", description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=2026, help="semente dos lances aleatórios")
    p.add_argument("--delay", type=float, default=0.05,
                   help="segundos entre lances; 0 mostra só o resultado final")
    p.add_argument("--max-plies", type=int, default=None, help="limite de meios-lances")
    p.add_argument("--fen", default=None, help="posição inicial (padrão: inicial do xadrez)")
    p.add_argument("--ascii", action="store_true", help="usa letras em vez de símbolos Unicode")
    p.add_argument("--svg", type=Path, help="salva a imagem do tabuleiro final")
    p.add_argument("--pgn", type=Path, help="salva a partida em PGN")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # peças Unicode no Windows
    if os.name == "nt":
        os.system("")                              # habilita sequências ANSI no console
    unicode = not args.ascii

    def show(state: ChessState) -> None:
        print(_CLEAR + render(state, unicode=unicode), flush=True)
        time.sleep(args.delay)

    reset_transition_count()
    try:
        state, outcome = play_random_game(args.seed, args.fen, args.max_plies,
                                          show if args.delay > 0 else None)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        raise SystemExit(130)

    if args.delay <= 0:
        print(render(state, unicode=unicode))
    print(f"\nResultado: {outcome.result.value} ({outcome.reason or 'em andamento'}) | "
          f"{len(state.move_stack)} meios-lances | {transition_count()} transições")
    if args.svg:
        args.svg.write_text(render_svg(state), encoding="utf-8")
        print(f"SVG salvo em {args.svg}")
    if args.pgn:
        args.pgn.write_text(to_pgn(state, {"Event": "Demonstração do motor",
                                           "White": "aleatório", "Black": "aleatório"}),
                            encoding="utf-8")
        print(f"PGN salvo em {args.pgn}")
    return outcome


if __name__ == "__main__":
    main()
