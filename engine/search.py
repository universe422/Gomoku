"""Gumbel root comparison and completed-Q targets over exact game states.

Equations: google-deepmind/mctx, _src/policies.py and _src/qtransforms.py.
Independent implementation; not the full AlphaZero system. Values are bounded
by [-1, 1], so tiny Q gaps are deliberately not min/max rescaled to [0, 1].
"""
from dataclasses import dataclass, field
import math
import random
from browser_support import torch
from core import *
from threat_search import root_choices
from forcing import four_moves, solve_forcing, immediate_foul_wins, immediate_foul_defences

SEARCH_VERSION = 3
# V2 completed-Q policy labels remain mathematically compatible. V1's
# discarded-candidate targets must still never be taught to this model.
POLICY_MIN_VERSION = 2


@dataclass
class Node:
    player: int
    prior: float = 1.
    visits: int = 0
    total: float = 0.
    raw_value: float = 0.
    solved: object = None
    complete: bool = True
    children: dict = field(default_factory=dict)
    trace: object = field(default=None, repr=False)

    @property
    def mean(self):
        if self.solved is not None:
            return self.solved
        return self.total / self.visits if self.visits else self.raw_value


def softmax(values):
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = sum(weights)
    return [v / total for v in weights]


def expand(node, g, net_id):
    choices, proof, _ = tactical(g)
    if proof is not None:
        node.solved = node.raw_value = proof
        actions = choices if proof == 1 else legal(g)
        node.children = {a: Node(3-node.player, 1/len(actions)) for a in actions}
        return proof
    actions = choices or legal(g)
    logits, value = yield (net_id, pack(g))
    node.raw_value = float(value)
    probs = softmax([float(logits[a]) for a in actions])
    node.children = {a: Node(3-node.player, max(p, 1e-30)) for a, p in zip(actions, probs)}
    return node.raw_value


def prove(node):
    if node.solved is not None or not node.children:
        return
    children = list(node.children.values())
    if any(c.solved == -1. for c in children):
        node.solved = 1.
    elif node.complete and all(c.solved is not None for c in children):
        node.solved = max(-c.solved for c in children)


def completed_scores(node):
    """Equal Q preserves the policy; unvisited actions keep completed values."""
    actions = list(node.children)
    children = [node.children[a] for a in actions]
    visited = [c for c in children if c.visits]
    n = sum(c.visits for c in visited)
    mass = sum(c.prior for c in visited)
    average = sum(c.prior * -c.mean for c in visited) / mass if mass else node.raw_value
    mixed = (node.raw_value + n * average) / (n + 1)
    q = [-c.mean if c.visits else mixed for c in children]
    scale = .1 * (50 + max((c.visits for c in children), default=0))
    scores = [math.log(c.prior) + scale * v for c, v in zip(children, q)]
    return actions, q, scores


def interior_action(node):
    actions, _, scores = completed_scores(node)
    winning = [a for a in actions if node.children[a].solved == -1.]
    if winning:
        return max(winning, key=lambda a: node.children[a].prior)
    viable = [a for a in actions if node.children[a].solved != 1.]
    allowed = set(viable or actions)
    probs = dict(zip(actions, softmax(scores)))
    n = sum(c.visits for c in node.children.values())
    return max((a for a in actions if a in allowed), key=lambda a:
               probs[a] - node.children[a].visits/(n+1))


def rollout(root, g, action, net_id):
    state = clone(g)
    node = root
    path = [node]
    move(state, action)
    node = node.children[action]
    path.append(node)
    while state.winner is None and node.children and node.solved is None:
        a = interior_action(node)
        move(state, a)
        node = node.children[a]
        path.append(node)
    if state.winner is not None:
        node.solved = outcome(state.winner, node.player)
        value = node.solved
    elif node.solved is not None:
        value = node.solved
    else:
        value = yield from expand(node, state, net_id)
    if root.trace is not None:
        depth = len(path)-1
        root.trace['depth_sum'] += depth
        root.trace['depth_max'] = max(root.trace['depth_max'], depth)
        root.trace['rollouts'] += 1
    for n in reversed(path):
        prove(n)
        if n.solved is not None:
            value = n.solved
        n.visits += 1
        n.total += value
        value = -value


def round_rollouts(root, g, actions, visits, net_id):
    """One coroutine per independent root branch, batched at neural leaves.

    Two simulations never edit the same child subtree concurrently. Allocation
    and Q estimates match the serial rounds when no proof ends search early.
    """
    pending = {}
    remaining = {a: visits for a in actions}
    used = 0

    def advance(a, gen=None, answer=None):
        nonlocal used
        while remaining[a] and root.solved is None:
            if gen is None:
                gen = rollout(root, g, a, net_id)
                try:
                    request = next(gen)
                except StopIteration:
                    remaining[a] -= 1
                    used += 1
                    gen = None
                    continue
            else:
                try:
                    request = gen.send(answer)
                except StopIteration:
                    remaining[a] -= 1
                    used += 1
                    gen = None
                    continue
            pending[a] = (gen, request)
            return

    for a in actions:
        advance(a)
    try:
        while pending and root.solved is None:
            batch = list(pending.items())
            pending.clear()
            # None is an internal batch marker; each nested request still names
            # its model. The parent process runs all leaves together on the GPU.
            answers = yield (None, [item[1][1] for item in batch])
            if len(answers) != len(batch):
                raise RuntimeError('Mismatched leaf batch')
            for (a, (gen, _)), answer in zip(batch, answers):
                if root.solved is not None:
                    gen.close()
                else:
                    advance(a, gen, answer)
    finally:
        for gen, _ in pending.values():
            gen.close()
    return used


def search(g, net_id=0, simulations=128, rng=None, selfplay=False,
           tactical_root=True, diagnostics=False, forcing=True, candidate_limit=0,
           forcing_nodes=64, forcing_plies=9):
    if simulations < 4:
        raise ValueError('simulations must be >=4')
    if g.winner is not None:
        raise ValueError('Finished board')
    if candidate_limit < 0 or forcing_nodes < 0 or forcing_plies < 0:
        raise ValueError('search limits must be nonnegative')
    import time
    started = time.perf_counter()
    rng = rng or random.Random()
    root = Node(g.turn)
    root.trace = {'depth_sum': 0, 'depth_max': 0, 'rollouts': 0}
    value = yield from expand(root, g, net_id)
    reason = 'mcts';certificate = None;initial = [];protected = []
    solver = {'status': 'not_run', 'nodes': 0, 'max_depth': 0, 'seconds': 0.}
    priors_before = {a: c.prior for a, c in root.children.items()}
    if tactical_root and root.solved is None:
        choices, reason = root_choices(g, wins, legal)
        if choices:
            root.children = {a: root.children[a] for a in choices}
            root.complete = False
            if reason == 'create-double-winning-threat':
                root.solved = 1.
    if tactical_root and forcing and root.solved is None and len(root.children)>1:
        if g.turn == BLACK and immediate_foul_wins(g):
            safe = set(immediate_foul_defences(g)) & root.children.keys()
            if safe:
                root.children = {a: c for a, c in root.children.items() if a in safe}
                root.complete = False
                reason = 'prevent-immediate-foul-trap'
        protected = [a for a, _ in four_moves(g) if a in root.children]
        if protected:
            direct_fouls = set(immediate_foul_wins(g)) & root.children.keys() if g.turn == WHITE else set()
            solver = solve_forcing(g, forcing_nodes, forcing_plies,
                {a: c.prior for a, c in root.children.items()}, direct_fouls or set(root.children))
            if solver['status'] == 'proven':
                if direct_fouls:
                    solver['actions'] = sorted(direct_fouls)
                certificate = solver
                root.children = {a: root.children[a] for a in solver['actions']}
                root.solved = 1.;root.complete = False;reason = 'verified-forcing-win'
    actions = list(root.children)

    def stats(used, count, q=None, p=None):
        result = {'candidates': count, 'used': used, 'value': root.mean,
                  'proof': root.solved, 'policy_weight': float(root.solved != -1.),
                  'search_version': SEARCH_VERSION, 'reason': reason,
                  'mcts_max_depth': root.trace['depth_max'],
                  'mcts_mean_depth': root.trace['depth_sum']/max(1,root.trace['rollouts']),
                  'forcing_nodes': solver['nodes'], 'forcing_max_depth': solver['max_depth'],
                  'forcing_seconds': solver['seconds'], 'elapsed_seconds': time.perf_counter()-started,
                  'protected_candidates': len(protected),
                  'protected_unsearched': len(set(protected)-set(initial)) if initial else 0,
                  'proof_plies': certificate['proof_plies'] if certificate else None}
        if certificate:
            result['certificate'] = certificate
        if q is not None:
            result['q_range'] = max(q)-min(q)
            result['chosen_target_probability'] = float(p[action])
        if diagnostics:
            all_actions, values, _ = completed_scores(root)
            ranked = sorted(priors_before, key=lambda a: (-priors_before[a], a))
            result.update({'actions': all_actions, 'q': values,
                'visits': [root.children[a].visits for a in all_actions],
                'priors': [root.children[a].prior for a in all_actions],
                'initial': initial, 'protected': protected,
                'policy_rank': {str(a): i+1 for i, a in enumerate(ranked)},
                'solved': [root.children[a].solved for a in all_actions]})
        return result

    if len(actions) == 1 or root.solved is not None:
        p = torch.zeros(ACTIONS);p[actions] = 1/len(actions)
        action = rng.choice(actions) if selfplay else actions[0]
        return action, p, stats(0,len(actions),p=p)
    noise_scale = (1. if g.stone_count < 24 else .25) if selfplay else 0.
    noise = {a: -math.log(-math.log(max(1e-12, rng.random()))) * noise_scale for a in actions}
    rank_prior = lambda a: math.log(root.children[a].prior)+noise[a]
    # More candidates require a larger budget; 128 still starts with 16.
    # Verified forcing moves get reserved slots instead of disappearing below
    # the neural top-K. Very small diagnostic budgets report any overflow.
    base = candidate_limit or (16 if simulations<256 else 32 if simulations<1024 else 64)
    capacity = min(len(actions), max(2,simulations//2))
    k = min(capacity, max(min(base,max(2,simulations//4)),len(protected)))
    reserved = sorted(protected,key=rank_prior,reverse=True)[:k]
    initial = reserved+[a for a in sorted(actions,key=rank_prior,reverse=True) if a not in reserved][:k-len(reserved)]
    active = initial[:];used = 0
    while len(active) > 1 and root.solved is None:
        rounds = math.ceil(math.log2(len(active)))
        visits = max(1,(simulations-used)//(len(active)*rounds))
        visits = min(visits,(simulations-used)//len(active))
        if visits < 1:
            break
        used += yield from round_rollouts(root,g,active,visits,net_id)
        all_actions,_,scores = completed_scores(root)
        rank = dict(zip(all_actions,scores))
        active.sort(key=lambda a: (root.children[a].solved == -1.,root.children[a].solved != 1.,rank[a]+noise[a]),reverse=True)
        active = active[:math.ceil(len(active)/2)]
    while used < simulations and root.solved is None:
        yield from rollout(root,g,active[0],net_id)
        used += 1
    all_actions,q,scores = completed_scores(root)
    winning = [a for a in all_actions if root.children[a].solved == -1.]
    p = torch.zeros(ACTIONS)
    if winning:
        p[winning] = 1/len(winning)
        action = max(winning,key=rank_prior)
    else:
        allowed = [a for a in all_actions if root.children[a].solved != 1.] or all_actions
        score_by_action = dict(zip(all_actions,scores))
        p[allowed] = torch.tensor(softmax([score_by_action[a] for a in allowed]))
        finalists = [a for a in active if a in allowed] or allowed
        action = max(finalists,key=lambda a: score_by_action[a]+noise[a])
    return action,p,stats(used,k,q,p)


def drive(generator, evaluator):
    try:
        request = next(generator)
    except StopIteration as end:
        return end.value
    while True:
        try:
            tag, payload = request
            if tag is None:
                answer = evaluator.batch(payload) if hasattr(evaluator, 'batch') else [evaluator(*r) for r in payload]
            else:
                answer = evaluator(tag, payload)
            request = generator.send(answer)
        except StopIteration as end:
            return end.value
