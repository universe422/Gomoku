"""Optional exclusive synchronous timers; never edit search/rule algorithms.

Only generator.next/send are timed as tree work. Awaited JS is inference_wait,
whose internals are separately measured by the worker and MUST NOT be added to
this total a second time. Every nested Python scope subtracts its children's
time. Diagnostics add instrumentation overhead; both comparison modes use it.
"""
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import sys
import time

enabled = False
timings = Counter()
calls = Counter()
counters = Counter()
stack = []
installed = False


@contextmanager
def scope(name):
    if not enabled:
        yield
        return
    frame = [time.perf_counter(), 0.0]
    stack.append(frame)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - frame[0]
        stack.pop()
        timings[name] += max(0.0, elapsed - frame[1]) * 1000
        calls[name] += 1
        if stack:
            stack[-1][1] += elapsed


def cache_info():
    import core, patterns, forcing
    return {name: fn.cache_info()._asdict() for name, fn in (
        ('legal', core._legal), ('wins', core._wins),
        ('bitboards', patterns.bitboards), ('fours', forcing._fours))}


cache_start = {}


def reset():
    global cache_start
    if stack:
        raise RuntimeError('Cannot reset metrics during an active operation')
    timings.clear(); calls.clear(); counters.clear()
    cache_start = cache_info()


def snapshot():
    now = cache_info()
    return {'timing_ms': dict(timings), 'calls': dict(calls), 'counters': dict(counters),
            'timing_semantics': 'Exclusive Python scopes. inference_wait includes worker/ORT time; do not add worker internals twice.',
            'rule_caches': {name: {**info,
                'operation_hits': info['hits']-cache_start.get(name,{}).get('hits',0),
                'operation_misses': info['misses']-cache_start.get(name,{}).get('misses',0)}
                for name,info in now.items()}}


def install():
    """Replace every imported alias to a measured function exactly once."""
    global installed
    if installed:
        return
    import core, renju, forcing, threat_search, search
    modules = [core, renju, forcing, threat_search, search]
    bridge = sys.modules.get('bridge')
    if bridge is not None:
        modules.append(bridge)
    categories = [
        ('rules', [core.legal, core.move, core.wins, renju.analyze_move]),
        ('tactical', [core.tactical, threat_search.root_choices, forcing.four_moves,
                      forcing.solve_forcing, forcing.immediate_foul_wins,
                      forcing.immediate_foul_defences]),
    ]
    for category, functions in categories:
        for function in functions:
            def wrap(fn, section):
                @wraps(fn)
                def measured(*args, **kwargs):
                    if not enabled:
                        return fn(*args, **kwargs)
                    with scope(section):
                        result = fn(*args, **kwargs)
                    if fn.__name__ == 'solve_forcing':
                        counters['tactical_proofs'] += int(result.get('status') == 'proven')
                        counters['tactical_unknowns'] += int(result.get('status') == 'unknown')
                    return result
                return measured
            measured = wrap(function, category)
            for module in modules:
                for name, value in list(vars(module).items()):
                    if value is function:
                        setattr(module, name, measured)
    installed = True
