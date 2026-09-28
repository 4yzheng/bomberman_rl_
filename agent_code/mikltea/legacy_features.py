"""
    Report features adapted from crazy_turtle, without model or PyTorch dependencies.

    To obtain features from a given game_state, call state_to_features
"""

import numpy as np
import copy
import settings


# ----------------------------------------<< utility functions >>-------------------------------------------------------

def look_for_targets(free_space, start, targets, allow_next_best=False):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, return None

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]

        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # Determine the first step towards the best found target tile
    if best not in targets and not allow_next_best:
        return None
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def get_reachable(free_space, start):
    """
    Find out which cells of the area are reachable from a given starting position using breadth-first search
    :param free_space: 2d - numpy array, where all free fields are 1, the rest is 0
    :param start: tuple containing starting-position
    :return: list of reachable positions
    """
    frontier = [start]
    parent_dict = {start: start}

    reachable = []

    while len(frontier) > 0:
        current = frontier.pop(0)
        x, y = current

        reachable.append(current)
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current

    return reachable  # , parent_dict


def position_valid(pos, shape):
    """
    return if pos lies in matrix with shape "shape"
    """
    return (0 <= pos[0] < shape[0]) and (0 <= pos[1] < shape[1])


def set_line(positions, explosion_map, field):
    """
    Draws a line (specified using positions) into the explosion map given the playing field.
    This is used to create custom explosion maps of the current state.

    :param positions: list of positions that specify the explosion of a bomb
    :param explosion_map: 2d-array indicating where explosions should be
    :param field: the field of the current game state
    """
    for pos in positions:
        if position_valid(pos, explosion_map.shape):
            if field[pos] != -1:
                explosion_map[pos] = 1
            else:
                break


def predict_explosion_map(bombs, field, query_times=[0]):
    """
    Generates a custom explosion map given certain parameters

    :param bombs: Bombs we want to predict the explosions for
    :param field: The arena containing crates and walls
    :param query_times: The time steps (relative to current time) in which the bombs explode.
        if query_time==[0], it means that the bombs explode in the next step.
        if query_time==None, it will ignore the time.
    :return:
    """
    RADIUS = settings.BOMB_POWER

    explosion_map = np.zeros_like(field)

    if query_times is None:
        query_times = [None]

    for bomb in bombs:
        # bombs stored in A list of tuples
        #  ((x, y), t) of coordinates and countdowns for
        # all active bombs.
        bomb_pos, bomb_time = bomb
        for query_time in query_times:
            if bomb_time == query_time or query_time is None:
                explosion_map[bomb_pos] = 1

                right = [(bomb_pos[0] + i, bomb_pos[1]) for i in range(RADIUS + 1)]
                left = [(bomb_pos[0] - i, bomb_pos[1]) for i in range(RADIUS + 1)]
                top = [(bomb_pos[0], bomb_pos[1] - i) for i in range(RADIUS + 1)]
                bottom = [(bomb_pos[0], bomb_pos[1] + i) for i in range(RADIUS + 1)]

                set_line(right, explosion_map, field)
                set_line(left, explosion_map, field)
                set_line(top, explosion_map, field)
                set_line(bottom, explosion_map, field)

    return explosion_map


def get_safe_field_positions(game_state):
    """
    Get the fields which don't lie in a blast radius

    :param game_state: Current game state
    :return: 2d- array specifying which field are safe (=1) or not (=0)
    """
    free = get_free_field_positions(game_state)

    field = game_state["field"]
    bombs = game_state["bombs"]

    explosion_map = predict_explosion_map(bombs, field, query_times=None)
    safe = np.multiply(free, 1 - explosion_map)
    return safe


def get_free_field_positions(game_state, include_self=False, include_others=True, include_bombs=True,
                             include_explosions=True):
    """
    returns binary matrix of field where free positions are 1, not free 0

    :param game_state: The current state of the game
    :param include_self: Flag whether the agent should be included as a free field or not.
            If this flag is true, then the agent's field is not considered free
    :param include_others: Flag whether the other agents should be included as a free field or not.
            If this flag is true, then the agent's field is not considered free
    :param include_bombs:  Flag whether the bombs should be included as a free field or not.
            If this flag is true, then the bomb's field is not considered free
    :param include_explosions: Flag whether the explosions should be included as a free field or not.
            If this flag is true, then the explosion's field is not considered free
    :return: 2d- array where free positions are 1, not free 0
    """

    field = game_state["field"]
    bombs = game_state["bombs"]
    free_space = field == 0

    if include_explosions:
        explosion_fire = 1 - game_state["explosion_map"]
        explosion_next = 1 - predict_explosion_map(bombs, field, [0])

        free_space = np.minimum(free_space, explosion_next)
        free_space = np.minimum(free_space, explosion_fire)

    if include_others:
        others = [xy for (n, s, b, xy) in game_state['others']]
        for o in others:
            free_space[o] = False

    if include_self:
        my_state = game_state["self"]
        _, _, _, my_pos = my_state
        free_space[my_pos] = False

    if include_bombs:
        for bomb_xy, t in game_state["bombs"]:
            free_space[bomb_xy] = False

    return free_space


def predict_destroyed_enemies(bomb_pos, game_state):
    """
    returns how many enemies are in the blast radius if a bomb were placed at bomb_pos

    :param bomb_pos: The position of the (hypothetical) bomb
    :param game_state: The current game state
    :return: number of enemies in the blast radius
    """

    field = game_state["field"]
    others = ([xy for (n, s, b, xy) in game_state['others']])
    explosion_map = predict_explosion_map([(bomb_pos, 0)], field, [0])
    count = 0
    for (x, y) in others:
        if explosion_map[x, y] == 1:
            count += 1

    return count


def predict_destroyed_crates(bomb_pos, field, query_time):
    """
    returns how many crates are in the blast radius if a bomb were placed at bomb_pos

    :param bomb_pos: The position of the (hypothetical) bomb
    :param field: The field the current state
    :param query_time: How many steps the bomb has left until it explodes
    :return: number of crates in the blast radius
    """

    explosion_map = predict_explosion_map([(bomb_pos, 0)], field, query_time)
    crates_field = copy.deepcopy(field)
    crates_field[crates_field != 1] = 0
    destroyed_crates = np.multiply(explosion_map, crates_field)

    destroyed_crates_count = np.count_nonzero(destroyed_crates)
    return destroyed_crates, destroyed_crates_count


def get_potential_traps(free, field):
    traps = np.zeros_like(field)
    for y, el in enumerate(free):
        for x, f_ in enumerate(el):
            if free[x, y] == 1:
                neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free[x, y] == 1]
                if len(neighbors) == 1:
                    traps[x, y] = 1
    return traps


def get_crate_score(pfree, pplayer, game_state):
    field = game_state["field"]
    _, destroyed_crate_count = predict_destroyed_crates(pfree, field, [0])
    killed_enemy_count = predict_destroyed_enemies(pfree, game_state)

    dist_score = np.linalg.norm(np.array(pfree) - np.array(pplayer), 1)

    if killed_enemy_count == 0:
        return destroyed_crate_count - 1 * dist_score
    return 5 * killed_enemy_count - 1 * dist_score


# ----------------------------------------<< feature computations >>----------------------------------------------------

def get_agent_state_feature(game_state):
    """
    returns the agent-feature f_agent

    Pseudocode -------------------------
    if agent in blast radius:
        f_agent = 0
    else:
        if number of reachable coins != 0:
            f_agent = 1
        else:
            f_agent = 2

    ------------------------------------
    :param game_state: Current game state
    :return: f_agent feature
    """
    my_state = game_state["self"]
    _, _, _, my_pos = my_state

    field = game_state["field"]
    bombs = game_state["bombs"]
    visible_coins = game_state['coins']

    explosion_future = predict_explosion_map(bombs, field, None)

    free = get_free_field_positions(game_state, include_explosions=False)

    reachable_coins = look_for_targets(free, my_pos, visible_coins)

    if explosion_future[my_pos] != 0:
        return 0

    if reachable_coins is not None:
        return 1

    return 2


def get_bomb_feature(game_state):
    """
    returns the bomb-feature f_bomb

    Pseudocode -------------------------
    if agent cannot place bomb:
        f_bomb = 0
    else:
        if placing bomb leads to being trapped:
            f_bomb = 1
        else:
            if placing bomb hits target:
                f_bomb = 2
            else:
                f_bomb = 1

    ------------------------------------

    :param game_state:
    :return:
    """
    my_state = game_state["self"]
    _, _, can_place_bomb, my_pos = my_state

    field = game_state["field"]
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]

    # --- find nearest target ---
    if not can_place_bomb:
        # print("exit at find nearest target")
        return 0

    # --- bad bomb placement ---
    free = get_free_field_positions(game_state)
    reachable_fields = get_reachable(free, my_pos)
    blast = predict_explosion_map([(my_pos, 0)], field)
    # print(blast)
    blast_pos = set(tuple(map(tuple, np.argwhere(blast == 1))))
    if len(set(reachable_fields) - blast_pos) == 0:
        # print("exit at bad bomb placement 1")
        return 1

    # --- bad bomb placement ---
    others = set([xy for (n, s, b, xy) in game_state['others']])
    crates_field = copy.deepcopy(field)
    crates_field[crates_field != 1] = 0
    crates_pos = set(tuple(map(tuple, np.argwhere(crates_field == 1))))
    if len(blast_pos.intersection(others)) == 0 and len(blast_pos.intersection(crates_pos)) == 0:
        # print("exit at bad bomb placement 2", others, blast_pos, crates_pos)
        return 1

    # --- good bomb ---
    return 2


def get_target_direction_feature(game_state):
    """
    returns the features f_target, f_coin, f_safe.
    Each of which indicate the direction the agent should take in order to reach the
    desired goal (target, coin, safety) the quickest.

    :param game_state: Current game state
    :return: features f_target, f_coin, f_safe
    """
    dir2idx = {
        (0, -1): 0,  # top
        (-1, 0): 1,  # left
        (0, 1): 2,  # bottom
        (1, 0): 3,  # right
        (0, 0): 4  # no direction
    }
    my_state = game_state["self"]
    field = game_state["field"]
    _, _, _, my_pos = my_state
    free = get_free_field_positions(game_state)

    # ------------------------< compute: nearest safe spot >------------------------
    nearest_safe_spot = -1  # default value: nothing found yet

    safe = get_safe_field_positions(game_state)
    safe_pos = tuple(map(tuple, np.argwhere(safe == 1)))
    reachable = look_for_targets(free, my_pos, safe_pos)
    if reachable is not None:
        direction = np.array(reachable) - np.array(my_pos)
        nearest_safe_spot = dir2idx[tuple(direction)]

    # ------------------------< compute: nearest coin >------------------------
    nearest_coin = -1  # default value: nothing found yet

    coins = game_state['coins']
    free_c = free

    # -- track special case: coins are blocked by an explosion --
    blocked_coins = []
    explosion_fire = game_state["explosion_map"]
    for coin in coins:
        if explosion_fire[coin] != 0:
            blocked_coins.append(True)
        else:
            blocked_coins.append(False)

    coins_blocked = all(blocked_coins)

    if coins_blocked:
        free_c = get_free_field_positions(game_state, include_explosions=False)

    # calculate direction to nearest reachable position of coin
    reachable = look_for_targets(free_c, my_pos, coins)
    if reachable is not None:
        direction = np.array(reachable) - np.array(my_pos)

        # if the proposed position lies in an explosion field, we should wait
        proposed_position = tuple(np.array(my_pos) + direction)

        if explosion_fire[proposed_position] == 1:
            nearest_coin = 4
        else:
            nearest_coin = dir2idx[tuple(direction)]

    # ------------------------< compute: best crate-bomb placement >------------------------
    nearest_crate = -1  # default value: nothing found yet

    reachable_fields = get_reachable(free, my_pos)
    reachable_targets = {}

    # --- first we add only the neighbors of reachable crates ---
    for (x, y) in reachable_fields:
        neighbors = tuple((x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if field[x, y] == 1)
        if len(neighbors) > 0:
            reachable_targets[(x, y)] = set()
            reachable_targets[(x, y)].update(neighbors)

    # --- now we add the targets near enemies ---
    if len(game_state['others']) != 0:
        others = ([xy for (n, s, b, xy) in game_state['others']])
        for (x, y) in others:
            for i in range(1, 4):
                neighbors = [(x + i, y), (x - i, y), (x, y - i), (x, y + i)]
                for (nx, ny) in neighbors:
                    if (nx, ny) in reachable_fields:
                        reachable_targets[(nx, ny)] = set()

    # --- now compute the field with the best score ---
    if len(reachable_targets.keys()) > 0:

        best_score = -float("inf")
        best_pos = None
        for free_pos in reachable_targets.keys():

            score = get_crate_score(free_pos, my_pos, game_state)
            if score > best_score:
                best_score = score
                best_pos = free_pos

        if best_pos == None:
            print("BUG ", best_pos, reachable_targets)
        next_step = look_for_targets(free, my_pos, [best_pos])
        direction = np.array(next_step) - np.array(my_pos)
        nearest_crate = dir2idx[tuple(direction)]

    return nearest_crate, nearest_coin, nearest_safe_spot



def state_to_features(game_state):
    """Return (mode, bomb_quality, target_direction, coin_direction, safe_direction)."""
    return (get_agent_state_feature(game_state), get_bomb_feature(game_state),
            *get_target_direction_feature(game_state))
