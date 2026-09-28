"""Three features: mode, suggested direction and useful-bomb flag."""
from collections import deque
import numpy as np
import settings

DIRECTIONS = ((0, -1), (-1, 0), (0, 1), (1, 0), (0, 0))
# Search order determines which equally short path is chosen.
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def find_step(free, start, targets):
    """Find the first step on a shortest path to any target."""
    if not targets:
        return None
    targets = set(targets)
    queue = deque([start])
    parents = {start: start}
    while queue:
        position = queue.popleft()
        if position in targets:
            while parents[position] != start:
                position = parents[position]
            return position
        x, y = position
        for dx, dy in NEIGHBORS:
            neighbor = (x + dx, y + dy)
            if free[neighbor] and neighbor not in parents:
                parents[neighbor] = position
                queue.append(neighbor)
    return None


def reachable_tiles(free, start):
    """Return reachable tiles in breadth-first order."""
    queue = deque([start])
    seen = {start}
    tiles = []
    while queue:
        position = queue.popleft()
        tiles.append(position)
        x, y = position
        for dx, dy in NEIGHBORS:
            neighbor = (x + dx, y + dy)
            if free[neighbor] and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return tiles


def blast_map(bombs, field):
    """Mark all blast tiles. Walls stop blasts; crates do not."""
    blast = np.zeros_like(field)
    for (x, y), timer in bombs:
        blast[x, y] = 1
        for dx, dy in NEIGHBORS:
            for distance in range(settings.BOMB_POWER + 1):
                nx, ny = x + dx * distance, y + dy * distance
                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                    break
                if field[nx, ny] == -1:
                    break
                blast[nx, ny] = 1
    return blast


def free_tiles(game, avoid_blasts=True):
    free = game['field'] == 0
    if avoid_blasts:
        free = np.minimum(free, 1 - game['explosion_map'])
        free = np.minimum(free, blast_map(game['bombs'], game['field']) == 0)
    for other in game['others']:
        free[other[3]] = False
    for position, timer in game['bombs']:
        free[position] = False
    return free


def direction(start, step):
    if step is None:
        return -1
    return DIRECTIONS.index((step[0] - start[0], step[1] - start[1]))


def coin_direction(game):
    free = free_tiles(game)
    if all(game['explosion_map'][coin] != 0 for coin in game['coins']):
        free = free_tiles(game, avoid_blasts=False)
    step = find_step(free, game['self'][3], game['coins'])
    if step is not None and game['explosion_map'][step] == 1:
        return 4
    return direction(game['self'][3], step)


def bomb_score(position, game):
    blast = blast_map([(position, 0)], game['field'])
    crates = int(((game['field'] == 1) & (blast != 0)).sum())
    enemies = sum(bool(blast[other[3]]) for other in game['others'])
    start = game['self'][3]
    distance = abs(position[0] - start[0]) + abs(position[1] - start[1])
    return (5 * enemies if enemies else crates) - distance


def target_direction(game):
    start = game['self'][3]
    free = free_tiles(game)
    reachable = reachable_tiles(free, start)
    targets = []
    for x, y in reachable:
        if any(game['field'][x + dx, y + dy] == 1 for dx, dy in NEIGHBORS):
            targets.append((x, y))
    for other in game['others']:
        x, y = other[3]
        for distance in range(1, 4):
            for position in ((x + distance, y), (x - distance, y), (x, y - distance), (x, y + distance)):
                if position in reachable and position not in targets:
                    targets.append(position)
    if not targets:
        return -1
    best = targets[0]
    best_score = bomb_score(best, game)
    for target in targets[1:]:
        score = bomb_score(target, game)
        if score > best_score:
            best, best_score = target, score
    return direction(start, find_step(free, start, [best]))


def useful_bomb(game):
    if not game['self'][2]:
        return False
    start = game['self'][3]
    blast = blast_map([(start, 0)], game['field'])
    reachable = reachable_tiles(free_tiles(game), start)
    if not any(blast[position] == 0 for position in reachable):
        return False
    hits_crate = ((game['field'] == 1) & (blast != 0)).any()
    hits_enemy = any(blast[other[3]] for other in game['others'])
    return bool(hits_crate or hits_enemy)


def escape_direction(game):
    """Search positions over time until reaching a tile outside pending blasts."""
    start = game['self'][3]
    occupied = {position for position, timer in game['bombs']}
    occupied.update(other[3] for other in game['others'])
    blasts, threatened, horizon = [], set(), 2
    for position, timer in game['bombs']:
        cells = {tuple(cell) for cell in np.argwhere(blast_map([(position, timer)], game['field']))}
        blasts.append((cells, timer + 1))
        threatened.update(cells)
        horizon = max(horizon, timer + 3)
    queue = deque([(start, 0, 4)])
    seen = {(start, 0)}
    while queue:
        position, time, first_move = queue.popleft()
        if position not in threatened and not game['explosion_map'][position]:
            return first_move
        if time >= horizon:
            continue
        for move, (dx, dy) in enumerate(DIRECTIONS):
            neighbor = (position[0] + dx, position[1] + dy)
            if game['field'][neighbor] != 0:
                continue
            if neighbor in occupied and not neighbor == position == start:
                continue
            arrival = time + 1
            if (neighbor, arrival) in seen or game['explosion_map'][neighbor] >= arrival:
                continue
            if any(neighbor in cells and detonation <= arrival < detonation + 2 for cells, detonation in blasts):
                continue
            seen.add((neighbor, arrival))
            queue.append((neighbor, arrival, move if time == 0 else first_move))
    return -1


def state_to_features(game):
    start = game['self'][3]
    if blast_map(game['bombs'], game['field'])[start]:
        mode, move = 0, escape_direction(game)
    elif find_step(free_tiles(game, avoid_blasts=False), start, game['coins']) is not None:
        mode, move = 1, coin_direction(game)
    else:
        mode, move = 2, target_direction(game)
    return mode, move, useful_bomb(game)
