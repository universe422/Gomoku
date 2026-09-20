import json
import time
from core import RenjuGame, legal, move, pack, PASS, BLACK
from search import search

game = RenjuGame()


def snapshot():
    actions = legal(game)
    forbidden = list(game.forbidden_moves()) if game.turn == BLACK and game.winner is None else []
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
    start = time.monotonic()
    generator = search(game, simulations=int(simulations), selfplay=False)
    evaluated = 0
    try:
        request = next(generator)
        while True:
            tag, payload = request
            batch = payload if tag is None else [(tag, payload)]
            inputs = [[list(p[0]), p[1], p[2]] for _, p in batch]
            answers = json.loads(await inferBatch(json.dumps(inputs)))
            evaluated += len(batch)
            reportProgress(evaluated)
            request = generator.send(answers if tag is None else answers[0])
    except StopIteration as end:
        action, _, stats = end.value
    finally:
        generator.close()
    move(game, action)
    return json.dumps(dict(state=snapshot(), stats=stats, seconds=time.monotonic()-start))
