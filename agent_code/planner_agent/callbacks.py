"""A small deterministic policy: survive, collect coins, then bomb useful targets."""
from collections import deque

import numpy as np
import settings as s

MOVES = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1),
         'LEFT': (-1, 0), 'WAIT': (0, 0)}


def setup(self):
    self.history = deque(maxlen=8)


def blast(field, origin):
    """Engine semantics: walls stop blasts; crates and bombs do not."""
    cells = {origin}
    for dx, dy in list(MOVES.values())[:4]:
        for distance in range(1, s.BOMB_POWER + 1):
            p = origin[0] + dx * distance, origin[1] + dy * distance
            if not (0 <= p[0] < field.shape[0] and 0 <= p[1] < field.shape[1]):
                break
            if field[p] == -1:
                break
            cells.add(p)
    return cells


class Forecast:
    """Known bombs only: no chain reactions exist in this engine.

    Index t means after t actions. A bomb at timer zero explodes after action 1.
    Crates/bombs cleared at t only become traversable on action t+1.
    """
    def __init__(self, state, plant=False):
        field = state['field']
        start = state['self'][3]
        bombs = list(state['bombs'])
        if plant:
            # A freshly placed bomb is decremented on this action already.
            bombs.append((start, s.BOMB_TIMER - 1))
        self.horizon = max([s.EXPLOSION_TIMER] +
                           [timer + 1 + s.EXPLOSION_TIMER for _, timer in bombs])
        shape = (self.horizon + 1,) + field.shape
        self.danger = np.zeros(shape, dtype=bool)
        cleared = np.full(field.shape, np.inf)
        blocked = np.zeros(field.shape, dtype=int)
        for t in range(1, self.horizon + 1):
            self.danger[t] = state['explosion_map'] >= t
        for p, timer in bombs:
            detonate = timer + 1
            blocked[p] = detonate
            for q in blast(field, p):
                self.danger[detonate:detonate + s.EXPLOSION_TIMER, q[0], q[1]] = True
                if field[q] == 1:
                    cleared[q] = min(cleared[q], detonate)
        self.open = np.zeros(shape, dtype=bool)
        for t in range(self.horizon + 1):
            self.open[t] = ((field == 0) | (cleared < t)) & (blocked < t)
            for other in state['others']:
                self.open[t][other[3]] = False
        self.start = start
        self.blocked = blocked
        # Backwards reachability: every allowed action has an entire escape path.
        self.alive = np.zeros(shape, dtype=bool)
        self.alive[-1] = self.open[-1] & ~self.danger[-1]
        for t in range(self.horizon - 1, -1, -1):
            next_safe = self.open[t + 1] & ~self.danger[t + 1] & self.alive[t + 1]
            reachable = next_safe.copy()  # WAIT
            reachable[1:] |= next_safe[:-1]
            reachable[:-1] |= next_safe[1:]
            reachable[:, 1:] |= next_safe[:, :-1]
            reachable[:, :-1] |= next_safe[:, 1:]
            # Waiting on an occupied bomb is legal, re-entering it is not.
            if blocked[start] >= t + 1 and not self.danger[t + 1][start]:
                reachable[start] |= self.alive[t + 1][start]
            self.alive[t] = reachable & ~self.danger[t]

    def routes(self, plant=False):
        """Earliest safe arrival at each tile, including waiting for explosions."""
        initial_time = int(plant)
        if plant and (self.danger[1][self.start] or not self.alive[1][self.start]):
            return {}
        queue = deque([(self.start, initial_time, 'WAIT')])
        seen = {(self.start, initial_time)}
        routes = {}
        while queue:
            p, time, first = queue.popleft()
            if self.alive[min(time, self.horizon)][p]:
                routes.setdefault(p, (time, first))
            for action, (dx, dy) in MOVES.items():
                q = p[0] + dx, p[1] + dy
                t = min(time + 1, self.horizon)
                if not (0 <= q[0] < self.open.shape[1] and 0 <= q[1] < self.open.shape[2]):
                    continue
                staying_on_bomb = q == p == self.start and self.blocked[q] >= t
                if not (self.open[t][q] or staying_on_bomb):
                    continue
                if self.danger[t][q] or not self.alive[t][q] or (q, t) in seen:
                    continue
                seen.add((q, t))
                queue.append((q, time + 1, action if time == initial_time else first))
        return routes


def act(self, game_state):
    if game_state['step'] == 1:
        self.history.clear()
    field = game_state['field']
    _, _, can_bomb, start = game_state['self']
    forecast = Forecast(game_state)
    routes = forecast.routes()
    self.history.append(start)
    threatened = set().union(*(blast(field, p) for p, _ in game_state['bombs']))
    quiet = [(p, route) for p, route in routes.items()
             if p not in threatened and not forecast.danger[:, p[0], p[1]].any()]
    # Leave blast lanes early: opponents can move into a planned escape route.
    if (start in threatened or forecast.danger[1][start]) and quiet:
        return min(quiet, key=lambda item: item[1][0])[1][1]
    coins = [p for p in game_state['coins'] if p in routes]
    if coins:
        target = min(coins, key=lambda p: routes[p][0])
        if target != start:
            return routes[target][1]

    # Prefer nearby, productive bomb sites, using actual walking distance.
    enemies = {other[3] for other in game_state['others']}
    targets = []
    for p, (distance, action) in routes.items():
        cells = blast(field, p)
        crates = sum(field[q] == 1 and q not in threatened for q in cells)
        value = crates + 3 * len(cells & enemies)
        if not value:
            continue
        if p == start:
            if not can_bomb or start in threatened:
                continue
            escape = Forecast(game_state, plant=True).routes(plant=True)
            if not escape:
                continue
            action = 'BOMB'
        score = value / (distance + 4) - 0.03 * self.history.count(p)
        targets.append((score, -distance, action))
    if targets:
        return max(targets)[2]

    # No useful target: reach a quiet tile and avoid short loops.
    if quiet:
        p, (_, action) = min(quiet, key=lambda item: item[1][0] +
                             0.1 * self.history.count(item[0]))
        return action
    # If no escape exists, at least avoid dying on the very next action.
    for action, (dx, dy) in MOVES.items():
        q = start[0] + dx, start[1] + dy
        if forecast.open[1][q] and not forecast.danger[1][q]:
            return action
    return 'WAIT'
