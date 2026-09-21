"""Bounded, positive-only proofs for continuous-four attacks (VCF).

A defender's immediate win is checked BEFORE obliging it to block. Every
placement and forbidden point uses the unchanged authoritative Renju rules.
Failure to prove a win is UNKNOWN, never a loss or a draw training label.
This is not a complete solver: quiet preparations and continuous threes are
outside its search. Node and ply budgets also bound its CPU cost.
"""
from functools import lru_cache
from time import perf_counter
from core import BLACK, WHITE, PASS, analyze_move, clone, move, pack, unpack, wins
from patterns import fork_pairs


@lru_cache(maxsize=8192)
def _fours(board, player):
    g = unpack((board, player, 0))
    result = []
    for action, endpoints in fork_pairs(board, player).items():
        r, c = divmod(action, 15)
        status = analyze_move(g.board, r, c, player)
        if not status.legal:
            continue
        g.board[r][c] = player
        try:
            winning = tuple(sorted(a for a in endpoints
                if analyze_move(g.board, *divmod(a, 15), player).winner == player))
            if winning:
                result.append((action, winning))
        finally:
            g.board[r][c] = 0
    return tuple(sorted(result))


def four_moves(game, player=None):
    player = game.turn if player is None else player
    return _fours(pack(game)[0], player)


def immediate_foul_wins(game):
    """All white fours whose sole defence is forbidden, excluding black wins.

The hypothetical side is WHITE even when this is used to check a threat
before BLACK moves. Returned actions have not been placed on the real board.
"""
    if game.winner is not None:
        return {}
    result = {}
    for action, endpoints in four_moves(game, WHITE):
        if len(endpoints) != 1:
            continue
        h = clone(game); h.turn = WHITE
        move(h, action)
        if h.winner is not None or wins(h, BLACK):
            continue
        point = endpoints[0]
        status = analyze_move(h.board, *divmod(point, 15), BLACK)
        if not status.legal:
            result[action] = {'point': point, 'code': status.code}
    return result


def solve_forcing(game, max_nodes=64, max_plies=9, priorities=None, root_actions=None):
    """Return an exact winning certificate or status='unknown'.

`line` contains ONLY actual legal forcing moves / unique replies. A terminal
four's unavoidable reply and winning move are described by `finish`, not
invented as a principal variation. `proof_plies` includes those last 2 plies.
`max_depth` counts only placements actually examined by this solver.
"""
    start = perf_counter(); nodes = 0; max_depth = 0; exhausted = False
    priorities = priorities or {}; memo = {};depth_limit=max_plies

    def tick(depth):
        nonlocal nodes, max_depth, exhausted
        if nodes >= max_nodes:
            exhausted = True
            return False
        nodes += 1; max_depth = max(max_depth, depth)
        return True

    def walk(g, depth, roots=None):
        nonlocal exhausted
        remaining = depth_limit-depth
        if remaining < 1:
            exhausted = True
            return None
        key = (pack(g), remaining)
        if roots is None and key in memo:
            return memo[key]
        own = wins(g, g.turn)
        if own:
            action = max(own, key=lambda a: priorities.get(a, 0.) if depth == 0 else -a)
            if roots is not None and action not in roots:
                own = [a for a in own if a in roots]
                if not own:
                    return None
                action = own[0]
            if not tick(depth+1):
                return None
            return {'line': [action], 'proof_plies': 1, 'finish': {'kind': 'five'}}
        enemy = wins(g, 3-g.turn)
        allowed = set(roots) if roots is not None else None
        # An attacker must first answer an existing opponent four. It is still
        # allowed to make a forcing four while blocking that threat.
        if enemy:
            if len(enemy) != 1 or not analyze_move(g.board, *divmod(enemy[0], 15), g.turn).legal:
                return None
            allowed = {enemy[0]} if allowed is None else allowed & {enemy[0]}
        choices = [(a, e) for a, e in four_moves(g) if allowed is None or a in allowed]
        choices.sort(key=lambda row: (-len(row[1]), -priorities.get(row[0], 0.) if depth == 0 else 0., row[0]))
        for action, endpoints in choices:
            if not tick(depth+1):
                break
            h = clone(g); move(h, action)
            if h.winner == g.turn:
                return {'line': [action], 'proof_plies': 1, 'finish': {'kind': 'five'}}
            if h.winner is not None or wins(h, h.turn):
                continue
            # A four is a proof only if its final win fits the requested depth.
            if remaining < 3:
                exhausted = True
                continue
            finish = None
            if len(endpoints) >= 2:
                finish = {'kind': 'double_threat', 'endpoints': list(endpoints)}
            else:
                block = endpoints[0]
                status = analyze_move(h.board, *divmod(block, 15), h.turn)
                if not status.legal:
                    finish = {'kind': 'forbidden_block', 'point': block, 'code': status.code}
            if finish is not None:
                return {'line': [action], 'proof_plies': 3, 'finish': finish}
            if remaining < 5:
                exhausted = True
                continue
            if not tick(depth+2):
                break
            move(h, endpoints[0])
            if h.winner is not None:
                continue
            child = walk(h, depth+2)
            if child is not None:
                result = {'line': [action, endpoints[0]]+child['line'],
                          'proof_plies': 2+child['proof_plies'], 'finish': child['finish']}
                if roots is None:
                    memo[key] = result
                return result
        return None

    proof = None
    if game.winner is None and max_nodes > 0 and max_plies > 0:
        # Search short wins across the root before spending the whole budget
        # on one highly ranked, long and unsuccessful forcing branch.
        limits=list(range(3,max_plies+1,2)) or [max_plies]
        for depth_limit in limits:
            proof = walk(game,0,root_actions)
            if proof or nodes>=max_nodes:break
    result = {'status': 'proven' if proof else 'unknown', 'nodes': nodes,
              'max_depth': max_depth, 'budget_exhausted': exhausted,
              'seconds': perf_counter()-start, 'max_nodes': max_nodes, 'max_plies': max_plies}
    if proof:
        result.update(proof)
        result['actions'] = [proof['line'][0]]
    return result


def immediate_foul_defences(game):
    """Exact avoidance of white's NEXT forbidden-block four, not a game value.

Every legal black reply is checked. A returned move can still lose to a
longer combination; callers must give these policy lessons value_weight=0.
"""
    from core import legal
    if game.turn != BLACK or not immediate_foul_wins(game):
        return []
    good = []
    for action in legal(game):
        h = clone(game); move(h, action)
        if h.winner == BLACK or (h.winner is None and not wins(h, WHITE) and not immediate_foul_wins(h)):
            good.append(action)
    return good
