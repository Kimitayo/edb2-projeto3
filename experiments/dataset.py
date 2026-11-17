"""
dataset.py - Posicoes do Lichess (Plano v2.0, secao 9.1 e 9.2).

Fluxo:
  1. le partidas padrao de um arquivo PGN do Lichess (.pgn ou .pgn.zst);
  2. de cada partida, guarda candidatos por contexto (C1..C6);
  3. escolhe 20 posicoes por contexto (10 calibracao + 10 teste), sem
     compartilhar partidas entre os dois conjuntos, com seed 2026;
  4. grava experiments/data/positions.csv e manifest.json.

Uso:
  python -m experiments.dataset build --pgn caminho/arquivo.pgn.zst \
      --max-games 20000 --obtained 2026-11-17 --terms "texto dos termos do site"

As funcoes que usam python-chess importam a biblioteca so quando chamadas.
A parte de selecao/divisao (select_positions) e Python puro.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SEED = 2026
PER_SET_PER_CONTEXT = 10
DATA_DIR = Path(__file__).parent / "data"

CONTEXTS = {
    ("opening", False): "C1", ("opening", True): "C2",
    ("middlegame", False): "C3", ("middlegame", True): "C4",
    ("endgame", False): "C5", ("endgame", True): "C6",
}
CONTEXT_NAMES = {
    "C1": "Abertura / nao tatica", "C2": "Abertura / tatica",
    "C3": "Meio-jogo / nao tatica", "C4": "Meio-jogo / tatica",
    "C5": "Final / nao tatica", "C6": "Final / tatica",
}


# ----------------------------------------------------------------------------
# Classificacao (definicoes operacionais da secao 9.1 do plano)
# ----------------------------------------------------------------------------
def classify_phase(board) -> str:
    """Final: <= 6 pecas que nao sejam reis/peoes (somando os dois lados).
    Abertura: numero do lance <= 10 e nao e final. Meio-jogo: o resto.

    Interpretacao adotada para "ate o 10o lance completo": board.fullmove_number <= 10
    (a posicao ocorre antes de terminar o 10o lance das pretas). Clara deve usar a mesma.
    """
    import chess
    others = sum(1 for p in board.piece_map().values()
                 if p.piece_type not in (chess.KING, chess.PAWN))
    if others <= 6:
        return "endgame"
    if board.fullmove_number <= 10:
        return "opening"
    return "middlegame"


def classify_tactical(board) -> bool:
    """Tatica: o jogador da vez esta em xeque OU tem ao menos uma captura legal."""
    if board.is_check():
        return True
    return any(board.is_capture(m) for m in board.legal_moves)


def context_of(board) -> tuple[str, bool, str]:
    phase = classify_phase(board)
    tactical = classify_tactical(board)
    return phase, tactical, CONTEXTS[(phase, tactical)]


def position_key(board) -> str:
    """Identifica a posicao (pecas, vez, roque, en passant), sem contadores de lance."""
    return " ".join(board.fen().split()[:4])


def is_excluded(board) -> bool:
    """Exclusoes do plano: terminal ou com reivindicacao de empate disponivel."""
    return board.is_game_over() or board.can_claim_draw()


# ----------------------------------------------------------------------------
# Leitura de PGN
# ----------------------------------------------------------------------------
def _open_pgn(path: Path):
    if path.suffix == ".zst":
        try:
            import io
            import zstandard
        except ImportError as e:
            raise SystemExit("Para ler .zst instale:  pip install zstandard  "
                             "(ou descompacte o arquivo antes)") from e
        raw = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw),
                                encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _h(*parts) -> int:
    """Hash deterministico (nao depende do PYTHONHASHSEED)."""
    data = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def collect_candidates(pgn_path: str | Path, max_games: int, seed: int = SEED):
    """Para cada partida padrao, guarda ate 1 candidato por contexto.

    O candidato de cada (partida, contexto) e o de menor hash(seed, partida, contexto, ply):
    escolha deterministica e sem favorecer o comeco da partida.
    """
    import chess
    import chess.pgn

    path = Path(pgn_path)
    candidates: list[dict] = []
    games_read = 0
    with _open_pgn(path) as handle:
        while games_read < max_games:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            h = game.headers
            if h.get("Variant", "Standard") != "Standard":
                continue
            games_read += 1
            site = h.get("Site", "")
            game_id = site.rsplit("/", 1)[-1] if site else f"game{games_read}"
            board = game.board()
            moves: list[str] = []
            best: dict[str, tuple[int, dict]] = {}
            for ply, move in enumerate(game.mainline_moves()):
                # `board` e a posicao ANTES do lance `ply`; guarda os lances que levaram ate ela.
                if ply >= 1 and not is_excluded(board):
                    phase, tactical, ctx = context_of(board)
                    score = _h(seed, game_id, ctx, ply)
                    if ctx not in best or score < best[ctx][0]:
                        best[ctx] = (score, dict(
                            game_id=game_id, ply=ply, context=ctx, phase=phase,
                            tactical=tactical, moves_uci=" ".join(moves),
                            fen=board.fen(), key=position_key(board),
                            side_to_move="white" if board.turn else "black",
                            white=h.get("White", ""), black=h.get("Black", ""),
                            source_site=site))
                board.push(move)
                moves.append(move.uci())
            candidates.extend(info for _, info in best.values())
    return candidates, games_read


# ----------------------------------------------------------------------------
# Selecao e divisao (Python puro)
# ----------------------------------------------------------------------------
class NotEnoughPositions(Exception):
    pass


def pool_of(game_id: str, seed: int = SEED) -> str:
    """Cada PARTIDA pertence a um unico conjunto -> nunca ha partida nos dois."""
    return "calibration" if _h(seed, "pool", game_id) % 2 == 0 else "test"


def select_positions(candidates: list[dict], seed: int = SEED,
                     per_set: int = PER_SET_PER_CONTEXT) -> list[dict]:
    """Escolhe `per_set` posicoes por contexto em cada conjunto (calibracao/teste).

    Regras: sem posicoes duplicadas, sem partida compartilhada entre conjuntos,
    ordem de escolha deterministica (hash com a seed).
    """
    selected: list[dict] = []
    seen_keys: set[str] = set()
    shortages: list[str] = []
    for ctx in sorted(CONTEXT_NAMES):
        for set_name in ("calibration", "test"):
            pool = [c for c in candidates
                    if c["context"] == ctx and pool_of(c["game_id"], seed) == set_name]
            pool.sort(key=lambda c: _h(seed, set_name, c["game_id"], c["key"]))
            taken = 0
            for c in pool:
                if c["key"] in seen_keys:
                    continue
                seen_keys.add(c["key"])
                taken += 1
                selected.append({**c, "set": set_name, "_idx": taken})
                if taken == per_set:
                    break
            if taken < per_set:
                shortages.append(f"{ctx}/{set_name}: {taken}/{per_set}")
    if shortages:
        raise NotEnoughPositions(
            "Poucas posicoes candidatas (" + "; ".join(shortages) +
            "). Use um PGN maior ou aumente --max-games.")
    for s in selected:
        s["position_id"] = f"{s['set'][:3]}_{s['context']}_{s.pop('_idx'):02d}"
    return selected


# ----------------------------------------------------------------------------
# Arquivos de saida e leitura
# ----------------------------------------------------------------------------
POSITION_COLUMNS = ["position_id", "set", "context", "phase", "tactical", "game_id", "ply",
                    "side_to_move", "moves_uci", "fen", "white", "black", "source_site"]


@dataclass
class Position:
    position_id: str
    set: str
    context: str
    moves_uci: str
    fen: str = ""

    def to_board(self):
        """Reconstroi o tabuleiro COM historico (o plano proibe usar so a FEN)."""
        import chess
        board = chess.Board()
        for uci in self.moves_uci.split():
            board.push_uci(uci)
        return board


def write_positions(selected: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=POSITION_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in sorted(selected, key=lambda s: s["position_id"]):
            w.writerow(s)


def load_positions(path: str | Path, set_name: str | None = None) -> list[Position]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [Position(r["position_id"], r["set"], r["context"], r["moves_uci"], r.get("fen", ""))
            for r in rows if set_name in (None, "both", r["set"])]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(pgn: str, max_games: int, obtained: str, terms: str, out_dir: Path = DATA_DIR) -> None:
    candidates, games_read = collect_candidates(pgn, max_games)
    selected = select_positions(candidates)
    write_positions(selected, out_dir / "positions.csv")
    counts: dict[str, int] = {}
    for s in selected:
        counts[f"{s['set']}/{s['context']}"] = counts.get(f"{s['set']}/{s['context']}", 0) + 1
    manifest = {
        "source": "Lichess Open Database (partidas padrao)",
        "source_url": "https://database.lichess.org/",
        "file": Path(pgn).name,
        "file_sha256": _sha256(Path(pgn)),
        "obtained": obtained,
        "terms_of_reuse": terms,
        "games_read": games_read,
        "candidates": len(candidates),
        "seed": SEED,
        "per_set_per_context": PER_SET_PER_CONTEXT,
        "selected": len(selected),
        "counts": counts,
        "game_ids": sorted({s["game_id"] for s in selected}),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"{len(selected)} posicoes gravadas em {out_dir / 'positions.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="gera positions.csv e manifest.json")
    b.add_argument("--pgn", required=True)
    b.add_argument("--max-games", type=int, default=20000)
    b.add_argument("--obtained", required=True, help="data em que o arquivo foi baixado (AAAA-MM-DD)")
    b.add_argument("--terms", required=True, help="termos de reutilizacao copiados do site do Lichess")
    a = ap.parse_args()
    build(a.pgn, a.max_games, a.obtained, a.terms)


if __name__ == "__main__":
    main()
