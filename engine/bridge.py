import json
import time
from array import array
from core import RenjuGame, legal, move, pack, PASS, BLACK
from search import search
import web_profile as profile

game = RenjuGame()
mode = 'optimized'
_packed = bytearray()


def configure(mode='optimized', profiling=False):
    if mode not in ('baseline', 'optimized'):
        raise ValueError('Unknown bridge mode')
    globals()['mode'] = mode
    profile.install()
    profile.enabled = bool(profiling)
    reset_metrics()
    return json.dumps({'mode': mode, 'profiling': profile.enabled})


def reset_metrics():
    profile.reset()


def metrics_json():
    return json.dumps(dict(mode=mode, **profile.snapshot()))


def clear_caches():
    import core, patterns, forcing
    for fn in (core._legal, core._wins, patterns.bitboards, forcing._fours):
        fn.cache_clear()
    reset_metrics()


def load_position(raw):
    """Developer fixtures only. Replay through the unchanged authoritative rules."""
    global game
    fixture = json.loads(raw)
    candidate = RenjuGame()
    if 'moves' in fixture:
        for action in fixture['moves']:
            if isinstance(action, bool) or not isinstance(action, int) or action not in legal(candidate):
                raise ValueError('Fixture contains an illegal action')
            move(candidate, action)
        if 'turn' in fixture and candidate.turn != fixture['turn']:
            raise ValueError('Fixture turn differs from its replay')
    else:
        # Synthetic edge fixtures used only by the regression runner.
        board = fixture['board']
        if len(board) != 225 or any(v not in (0, 1, 2) for v in board):
            raise ValueError('Invalid fixture board')
        candidate.board = [board[i:i+15] for i in range(0, 225, 15)]
        candidate.stone_count = sum(v != 0 for v in board)
        candidate.turn = fixture.get('turn', BLACK)
        if candidate.turn not in (1, 2):
            raise ValueError('Invalid fixture turn')
        candidate.passes = fixture.get('passes', 0)
        candidate.winner = fixture.get('winner')
        candidate.reason = fixture.get('reason', 'ok')
    game = candidate
    return json.dumps(snapshot())


def snapshot():
    with profile.scope('snapshot'):
        actions = legal(game)
        forbidden = []
        if game.turn == BLACK and game.winner is None and game.stone_count:
            if mode == 'baseline':
                forbidden = list(game.forbidden_moves())
            else:
                # Opening center restriction is not a foul. On every nonempty
                # unfinished black board legal() excludes exactly the fouls.
                allowed = set(actions)
                forbidden = [divmod(a, 15) for a, v in enumerate(pack(game)[0])
                             if v == 0 and a not in allowed]
        return dict(board=game.board, turn=game.turn, winner=game.winner,
                    reason=game.reason, count=game.stone_count, last=game.last_move,
                    forbidden=forbidden, canPass=actions == (PASS,))


def command(raw):
    global game
    message = json.loads(raw)
    if message['type'] == 'new':
        game = RenjuGame()
    elif message['type'] == 'move':
        action = int(message['action'])
        if action not in legal(game):
            if action == PASS:
                raise ValueError('둘 곳이 없을 때만 패스할 수 있습니다.')
            result = game.inspect(*divmod(action, 15))
            raise ValueError(result.message)
        move(game, action)
    return json.dumps(snapshot())


async def ai_turn(simulations):
    from js import inferBatch, reportProgress
    global _packed
    simulations = int(simulations)
    if not 4 <= simulations <= 16384:
        raise ValueError('simulations must be an integer in [4, 16384]')
    start = time.monotonic()
    generator = search(game, simulations=simulations, selfplay=False)
    evaluated = 0
    profile.counters['requested_budget'] = simulations
    try:
        with profile.scope('tree'):
            request = next(generator)
        while True:
            tag, payload = request
            batch = payload if tag is None else [(tag, payload)]
            if mode == 'baseline':
                with profile.scope('python_conversion'):
                    inputs = [[list(p[0]), p[1], p[2]] for _, p in batch]
                    raw = json.dumps(inputs)
                with profile.scope('inference_wait'):
                    output = await inferBatch(raw)
                with profile.scope('python_conversion'):
                    answers = json.loads(output)
            else:
                from js import inferPacked
                n = len(batch)
                with profile.scope('python_conversion'):
                    size = n * 227
                    if len(_packed) < size:
                        _packed = bytearray(size)
                    for i, (_, (board, turn, passed)) in enumerate(batch):
                        offset = i * 227
                        _packed[offset:offset+225] = board
                        _packed[offset+225] = turn
                        _packed[offset+226] = passed
                    view = memoryview(_packed)[:size]
                # One search at a time: the reusable bytearray remains untouched
                # until JS has copied/encoded its input and this await completes.
                try:
                    with profile.scope('inference_wait'):
                        output = await inferPacked(view, n)
                    with profile.scope('python_conversion'):
                        converted = output.to_py()
                        try:
                            # Deliberate copy into Python-owned storage; no
                            # borrowed JS buffer survives the next inference.
                            flat = array('f', converted)
                        finally:
                            if isinstance(converted, memoryview):
                                converted.release()
                        if len(flat) != size:
                            raise RuntimeError('Mismatched packed inference output')
                        answers = [(flat[i*227:i*227+226], flat[i*227+226]) for i in range(n)]
                        del output
                finally:
                    view.release()
            evaluated += len(batch)
            profile.counters['neural_requests'] += len(batch)
            profile.counters['progress_callbacks'] += 1
            reportProgress(evaluated)
            with profile.scope('tree'):
                request = generator.send(answers if tag is None else answers[0])
    except StopIteration as end:
        action, _, stats = end.value
    finally:
        generator.close()
    move(game, action)
    state = snapshot()
    profile.counters['completed_rollouts'] = stats['used']
    profile.counters['search_proved'] = int(stats['proof'] is not None)
    return json.dumps(dict(state=state, stats=stats, seconds=time.monotonic()-start,
                           metrics=dict(mode=mode, **profile.snapshot())))
