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

SEARCH_VERSION = 2


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
           tactical_root=True, diagnostics=False):
    if simulations < 4:
        raise ValueError('simulations must be >=4')
    if g.winner is not None:
        raise ValueError('Finished board')
    rng = rng or random.Random()
    root = Node(g.turn)
    value = yield from expand(root, g, net_id)
    if tactical_root and root.solved is None:
        choices, reason = root_choices(g, wins, legal)
        if choices:
            root.children = {a: root.children[a] for a in choices}
            root.complete = False
            if reason == 'create-double-winning-threat':
                root.solved = 1.
    actions = list(root.children)
    if len(actions) == 1 or root.solved is not None:
        p = torch.zeros(ACTIONS)
        p[actions] = 1/len(actions)
        action = rng.choice(actions) if selfplay else actions[0]
        return action, p, {'candidates': len(actions), 'used': 0, 'value': root.solved if root.solved is not None else value,
                          'proof': root.solved, 'policy_weight': float(root.solved != -1.), 'search_version': SEARCH_VERSION}
    # Keep one perturbation through all rounds. Training plays the finalist;
    # there is no second sampling step from shallow, eliminated candidates.
    noise_scale = (1. if g.stone_count < 24 else .25) if selfplay else 0.
    noise = {a: -math.log(-math.log(max(1e-12, rng.random()))) * noise_scale for a in actions}
    k = min(16, len(actions), max(2, simulations//4))
    initial = sorted(actions, key=lambda a: math.log(root.children[a].prior)+noise[a], reverse=True)[:k]
    active = initial[:]
    used = 0
    while len(active) > 1 and root.solved is None:
        rounds = math.ceil(math.log2(len(active)))
        visits = max(1, (simulations-used)//(len(active)*rounds))
        used += yield from round_rollouts(root, g, active, visits, net_id)
        all_actions, _, scores = completed_scores(root)
        rank = dict(zip(all_actions, scores))
        active.sort(key=lambda a: (root.children[a].solved == -1., root.children[a].solved != 1., rank[a]+noise[a]), reverse=True)
        active = active[:math.ceil(len(active)/2)]
    while used < simulations and root.solved is None:
        yield from rollout(root, g, active[0], net_id)
        used += 1
    all_actions, q, scores = completed_scores(root)
    winning = [a for a in all_actions if root.children[a].solved == -1.]
    p = torch.zeros(ACTIONS)
    if winning:
        p[winning] = 1/len(winning)
        action = max(winning, key=lambda a: math.log(root.children[a].prior)+noise[a])
    else:
        allowed = [a for a in all_actions if root.children[a].solved != 1.] or all_actions
        score_by_action = dict(zip(all_actions, scores))
        p[allowed] = torch.tensor(softmax([score_by_action[a] for a in allowed]))
        finalists = [a for a in active if a in allowed]
        # Prefer an untested option over a refuted loss if every finalist lost.
        if not finalists:
            finalists = allowed
        action = max(finalists, key=lambda a: score_by_action[a]+noise[a])
    stats = {'candidates': k, 'used': used, 'value': root.mean, 'proof': root.solved,
             'policy_weight': float(root.solved != -1.), 'search_version': SEARCH_VERSION,
             'q_range': max(q)-min(q), 'chosen_target_probability': float(p[action])}
    if diagnostics:
        stats.update({'actions': all_actions, 'q': q, 'visits': [root.children[a].visits for a in all_actions],
                      'priors': [root.children[a].prior for a in all_actions], 'initial': initial,
                      'solved': [root.children[a].solved for a in all_actions]})
    return action, p, stats


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
