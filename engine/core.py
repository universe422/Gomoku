"""Rules adapter and neural network. No absolute-coordinate policy table."""
from functools import lru_cache
import hashlib
from pathlib import Path
from renju import RenjuGame, BLACK, WHITE, EMPTY, SIZE, analyze_move
from threat_search import WINDOWS
from patterns import possible_black_fouls, winning_candidates

PASS=225
ACTIONS=226
FORMAT='renju-next-v1'
RULES_HASH=hashlib.sha256(Path(__file__).with_name('renju.py').read_bytes()).hexdigest()


def clone(g):
    h=RenjuGame();h.board=[row[:] for row in g.board]
    for k in ('turn','winner','reason','stone_count','passes','last_move'):setattr(h,k,getattr(g,k))
    return h


def move(g,a):
    r=g.pass_turn() if a==PASS else g.play(*divmod(a,15))
    if not r.legal:raise RuntimeError(f'Illegal action {a}: {r.message}')
    g.history.clear()


def outcome(winner,player):
    if winner is None:raise ValueError('Unfinished game')
    return 0. if winner==0 else 1. if winner==player else -1.


def pack(g):return (bytes(v for row in g.board for v in row),g.turn,int(g.passes>0))


def unpack(p):
    b,turn,passes=p;g=RenjuGame();g.board=[list(b[i:i+15]) for i in range(0,225,15)]
    g.turn=turn;g.passes=passes;g.stone_count=sum(v!=0 for v in b)
    return g


@lru_cache(maxsize=12000)
def _legal(b,turn):
    if not any(b):return (112,)
    if turn==WHITE or b.count(BLACK)<4:actions=tuple(i for i,v in enumerate(b) if v==0)
    else:
        suspects=possible_black_fouls(b);g=unpack((b,turn,0))
        actions=tuple(a for a,v in enumerate(b) if v==0 and
                      (a not in suspects or analyze_move(g.board,*divmod(a,15),BLACK).legal))
    return actions if actions else (PASS,)


def legal(g):return () if g.winner is not None else _legal(pack(g)[0],g.turn)


@lru_cache(maxsize=4096)
def _wins(b,player):
    if b.count(player)<4:return ()
    board=unpack((b,player,0)).board
    return tuple(a for a in winning_candidates(b,player)
                 if analyze_move(board,*divmod(a,15),player).winner==player)


def wins(g,player):return list(_wins(pack(g)[0],player))


def tactical(g):
    own=wins(g,g.turn)
    if own:return own,1.,'win'
    enemy=wins(g,3-g.turn)
    if len(enemy)==1 and enemy[0] in legal(g):return enemy,None,'block'
    if enemy:return None,-1.,'forced_loss' # own win checked first; cannot block two endpoints
    return None,None,'none'


