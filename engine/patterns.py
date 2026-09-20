"""Integer masks only prefilter patterns; the existing rule engine decides fouls.

All legal board locations remain available, including distant moves. A black
foul must first form an overline, two geometric fours or two geometric threes.
Recursive fake-three and exact-five precedence stay in renju.analyze_move.
"""
from functools import lru_cache

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def windows(length, open_ends=False):
    result = []
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for r in range(15):
            for c in range(15):
                rr, cc = r+(length-1)*dr, c+(length-1)*dc
                if not (0 <= rr < 15 and 0 <= cc < 15):
                    continue
                mask = sum(1 << ((r+i*dr)*15+c+i*dc) for i in range(length))
                ends = 0
                if open_ends:
                    er, ec, fr, fc = r-dr, c-dc, rr+dr, cc+dc
                    if not (0 <= er < 15 and 0 <= ec < 15 and 0 <= fr < 15 and 0 <= fc < 15):
                        continue
                    ends = (1 << (er*15+ec)) | (1 << (fr*15+fc))
                result.append((d, mask, ends))
    return tuple(result)


FIVE = windows(5)
SIX = windows(6)
OPEN_FOUR = windows(4, True)


@lru_cache(maxsize=4096)
def bitboards(board):
    black = white = 0
    for a, v in enumerate(board):
        if v == 1:
            black |= 1 << a
        elif v == 2:
            white |= 1 << a
    return black, white


def indices(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length()-1
        bits ^= low


def winning_candidates(board, player):
    black, white = bitboards(board)
    own, enemy = (black, white) if player == 1 else (white, black)
    candidates = 0
    for _, mask, _ in FIVE:
        if not enemy & mask and (own & mask).bit_count() == 4:
            candidates |= mask & ~own
    return indices(candidates)


def fork_pairs(board, player):
    black, white = bitboards(board)
    own, enemy = (black, white) if player == 1 else (white, black)
    if own.bit_count() < 3:
        return {}
    pairs = {}
    for _, mask, _ in FIVE:
        if not enemy & mask and (own & mask).bit_count() == 3:
            a, b = indices(mask & ~own)
            pairs.setdefault(a, set()).add(b)
            pairs.setdefault(b, set()).add(a)
    return pairs


def possible_black_fouls(board):
    own, enemy = bitboards(board)
    if own.bit_count() < 4:
        return frozenset()
    possible = set()
    if own.bit_count() >= 5:
        for _, mask, _ in SIX:
            if not enemy & mask and (own & mask).bit_count() == 5:
                possible.update(indices(mask & ~own))
    for patterns, count in [(FIVE, 3), (OPEN_FOUR, 2)]:
        groups = {}
        for d, mask, ends in patterns:
            stones = own & mask
            if enemy & mask or (own | enemy) & ends or stones.bit_count() != count:
                continue
            for a in indices(mask & ~(own | enemy)):
                shapes = groups.setdefault(a, set())
                shapes.add((d, stones | (1 << a)))
                if len(shapes) >= 2:
                    possible.add(a)
    return frozenset(possible)
