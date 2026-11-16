# Protocolo experimental

**Responsável:** Integrante 5 (Victor Alves). Este documento será detalhado pelo responsável.

As definições oficiais (orçamentos de tempo e de transições, métricas, posições do Lichess, referência Stockfish, calibração × teste e partidas completas) estão em [PLANO_PROJETO_XADREZ_EDBII.md](PLANO_PROJETO_XADREZ_EDBII.md), Seções 8 e 9.

Apoio do motor de xadrez para os experimentos (ver [game_specification.md](game_specification.md) e [guia_dev.md](../guia_dev.md)):
- contagem de transições: `reset_transition_count()` / `transition_count()`;
- reconstrução de posições com histórico: `from_moves(lances, start_fen)`;
- registro de estados e partidas: `serialize()` / `deserialize()` e `to_pgn()`.
