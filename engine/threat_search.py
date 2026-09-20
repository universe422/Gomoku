"""Exact shallow threat checks. These are search assistance, not learned skill.
Only applied at the MCTS root to bound additional CPU work.
"""
from renju import SIZE, EMPTY, BLACK, DIRECTIONS, inside, analyze_move
from collections import defaultdict
from patterns import fork_pairs

WINDOWS = tuple(tuple((r+i*dr,c+i*dc) for i in range(5))
                for r in range(SIZE) for c in range(SIZE) for dr,dc in DIRECTIONS
                if inside(r+4*dr,c+4*dc))


def fork_moves(game, player):
    """Legal moves creating >=2 distinct legal winning endpoints.

    Precondition: player currently has no immediate win. Each newly created
    winning endpoint must share a five-cell window with the new stone. A window
    with three stones and two empties identifies both the move and endpoint.
    All candidate moves and endpoints are checked by the authoritative rules.
    """
    board=game.board
    pairs=fork_pairs(bytes(v for row in board for v in row),player)
    forks={}
    for action,endpoints in pairs.items():
        if len(endpoints)<2:continue
        r,c=divmod(action,SIZE)
        if not analyze_move(board,r,c,player).legal:continue
        board[r][c]=player
        try:
            wins=[]
            for endpoint in endpoints:
                rr,cc=divmod(endpoint,SIZE)
                status=analyze_move(board,rr,cc,player)
                if status.legal and status.winner==player:wins.append(endpoint)
            if len(wins)>=2:forks[action]=sorted(wins)
        finally:board[r][c]=EMPTY
    return forks


def root_choices(game, immediate_wins, legal_actions=None):
    """Return root restriction and reason; None keeps existing MCTS choices.

    Preserve all immediate wins/blocks handled by v2. When facing a possible
    fork, retain verified fork-prevention moves and forcing counterattacks.
    Counterattacks are retained, not claimed to be winning/safe in all lines.
    """
    if game.winner is not None or game.stone_count<5:return None,'none'
    own=immediate_wins(game,game.turn)
    enemy=immediate_wins(game,3-game.turn)
    if own or enemy:return None,'immediate-handled-by-v2'
    own_forks=fork_moves(game,game.turn)
    if own_forks:return sorted(own_forks),'create-double-winning-threat'
    enemy_forks=fork_moves(game,3-game.turn)
    if not enemy_forks:return None,'none'
    allowed=[]
    actions=legal_actions(game) if legal_actions else [r*SIZE+c for r,c in game.legal_moves()]
    for action in actions:
        if action>=SIZE*SIZE:continue
        r,c=divmod(action,SIZE)
        game.board[r][c]=game.turn
        try:
            # A forcing four obliges the opponent to respond, so do not prune it.
            forcing=bool(immediate_wins(game,game.turn))
            if forcing or not fork_moves(game,3-game.turn):allowed.append(r*SIZE+c)
        finally:game.board[r][c]=EMPTY
    # If no answer exists, retain unrestricted search instead of an empty tree.
    return (allowed,'prevent-double-winning-threat') if allowed else (None,'no-shallow-answer')
