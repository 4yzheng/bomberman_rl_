"""Memoryless report features with small, independently testable refinements."""
from collections import deque
from . import legacy_features as base
from .config import CONFIG as C

STEPS = ((0, -1), (-1, 0), (0, 1), (1, 0), (0, 0))
_distances = {}


def distances(game, start):
    free = base.get_free_field_positions(game)
    found, queue = {start: 0}, deque([start])
    while queue:
        p = queue.popleft()
        for dx, dy in STEPS[:4]:
            q = p[0] + dx, p[1] + dy
            if free[q] and q not in found:
                found[q] = found[p] + 1
                queue.append(q)
    return found


def escape(game, plant=False):
    """BFS to a quiet tile, bounded by the known bombs' remaining lifetimes.

    Crates/opponents remain blocked; no speculative future terrain or enemy moves.
    The proposed bomb consumes the current action, leaving three moves to escape.
    """
    start = game['self'][3]
    bombs = list(game['bombs']) + ([(start, 3)] if plant else [])
    field = game['field']
    occupied = {p for p, _ in bombs} | {o[3] for o in game['others']}
    blasts = [(set(map(tuple, base.np.argwhere(base.predict_explosion_map([(p, t)], field, None)))), t + 1)
              for p, t in bombs]
    threatened = set().union(*(cells for cells, _ in blasts))
    horizon = max([2] + [t + 2 for _, t in blasts])
    first_time = int(plant)
    def safe(p, t):
        return game['explosion_map'][p] < t and not any(p in cells and det <= t < det + 2 for cells, det in blasts)
    if plant and not safe(start, 1):
        return -1
    queue, seen = deque([(start, first_time, 4)]), {(start, first_time)}
    fallback = -1
    while queue:
        p, t, direction = queue.popleft()
        if p not in threatened and not game['explosion_map'][p]:
            exits = sum(field[p[0]+dx, p[1]+dy] == 0 and (p[0]+dx, p[1]+dy) not in occupied
                        for dx, dy in STEPS[:4])
            if not C.get('shelter_exits') or exits >= 2:
                return direction
            if fallback == -1:
                fallback = direction
        if t >= horizon:
            continue
        for d, (dx, dy) in enumerate(STEPS):
            q = p[0] + dx, p[1] + dy
            if field[q] != 0 or (q in occupied and not q == p == start):
                continue
            if (q, t + 1) in seen or not safe(q, t + 1):
                continue
            seen.add((q, t + 1))
            queue.append((q, t + 1, d if t == first_time else direction))
    return fallback


def score(position, origin, game):
    blast = base.predict_explosion_map([(position, 0)], game['field'], None)
    crates = (game['field'] == 1) & (blast != 0)
    if C.get('ignore_doomed'):
        crates &= base.predict_explosion_map(game['bombs'], game['field'], None) == 0
    crate_count = int(crates.sum())
    enemy_count = 0
    for enemy in game['others']:
        p = enemy[3]
        if blast[p]:
            exits = sum(game['field'][p[0] + dx, p[1] + dy] == 0 for dx, dy in STEPS[:4])
            enemy_count += (1 if exits <= 2 else .25) if C.get('enemy_exits') else 1
    distance = _distances.get(position, 1000) if C.get('path_distance') else sum(abs(a-b) for a,b in zip(position, origin))
    enemy_value = C.get('enemy_weight', 5) * enemy_count
    value = (crate_count + enemy_value) if C.get('additive') else (enemy_value if enemy_count else crate_count)
    if C.get('ratio'):
        return value / (distance + C.get('overhead', 4))
    return value - C.get('distance_weight', 1) * distance


base.get_crate_score = score
_original_free = base.get_free_field_positions

def free_fields(game, *args, **kwargs):
    free = _original_free(game, *args, **kwargs)
    if C.get('safe_paths') and kwargs.get('include_explosions', True):
        free = base.np.minimum(free, base.predict_explosion_map(game['bombs'], game['field'], None) == 0)
    return free

base.get_free_field_positions = free_fields


def state_to_features(game):
    global _distances
    _distances = distances(game, game['self'][3]) if C.get('path_distance') or C.get('coin_radius') else {}
    mode = base.get_agent_state_feature(game)
    target, coin, safe = base.get_target_direction_feature(game)
    if mode == 1 and C.get('coin_radius'):
        if min([_distances.get(p, 1000) for p in game['coins']] or [1000]) > C['coin_radius']:
            mode = 2
    if C.get('safe_coins') and mode != 0:
        free = base.get_safe_field_positions(game)
        step = base.look_for_targets(free, game['self'][3], game['coins'])
        mode = 1 if step is not None else 2
        if step is not None:
            delta = tuple(a-b for a,b in zip(step, game['self'][3]))
            coin = STEPS.index(delta)
    direction = (safe, coin, target)[mode]
    good_bomb = base.get_bomb_feature(game) == 2
    if C.get('safe_bomb') and good_bomb:
        good_bomb = escape(game, plant=True) != -1
    if C.get('timed_escape') and mode == 0:
        direction = escape(game)
    if C.get('ignore_doomed') and good_bomb:
        origin = game['self'][3]
        blast = base.predict_explosion_map([(origin, 0)], game['field'], None)
        doomed = base.predict_explosion_map(game['bombs'], game['field'], None)
        good_bomb = bool(((game['field'] == 1) & (blast != 0) & (doomed == 0)).any() or
                         any(blast[o[3]] for o in game['others']))
    if C.get('trapped_move') and mode == 0 and direction == -1:
        free = base.get_free_field_positions(game)
        x, y = game['self'][3]
        direction = next((i for i, (dx, dy) in enumerate(STEPS[:4]) if free[x+dx, y+dy]), -1)
    return int(mode), int(direction), bool(good_bomb)
