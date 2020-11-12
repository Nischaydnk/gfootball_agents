import random

import math
import numpy as np
from kaggle_environments.envs.football.helpers import *

WING = 0.32
SAFE_DISTANCE = 0.085
POST = 0.045
SECTOR_SIZE = 0.2
LONG_SHOOTING_DISTANCE = 0.4
SHOOTING_DISTANCE = 0.36
SHORT_SHOOTING_DISTANCE = 0.24
FK_SHOOTING_DISTANCE = 0.3
HALF = 0
LAST_THIRD = 0.3
BRAKING_DISTANCE = 0.26
SLIDING_TACKLE_RANGE = 0.06
FIRST_TOUCH_TIMING = 1
FIELD_WIDTH = 0.42

PASS_SPEED = 0.015
RUNNING_SPEED = 0.01
DRIBBLING_SPEED = 0.01
SPRINTING_SPEED = 0.02

SLIDING_TACKLE_MODE = 0
USE_MODELS = False


# LOGGERS

def print_sticky_actions(obs):
    return ','.join([action.name for action in obs['sticky_actions']])


# HELPERS

def solve_eq(l1, l2a, l2b):
    x = 1
    solutions = []
    while x > -1:
        y1 = l1(x)
        y2a = l2a(x)
        y2b = l2b(x)
        if abs(y1) < FIELD_WIDTH and ((abs(y2a) < FIELD_WIDTH and abs(y1 - y2a) < 0.005) or
                                      (abs(y2b) < FIELD_WIDTH and abs(y1 - y2b) < 0.005)):
            solutions.append((x, y1))
            if len(solutions) == 2:
                return solutions
        x -= 0.0025
    return solutions


def custom_convert_observation(observation, fixed_positions):
    def do_flatten(obj):
        """Run flatten on either python list or numpy array."""
        if type(obj) == list:
            return np.array(obj).flatten()
        return obj.flatten()

    final_obs = []
    for obs in observation:
        o = []
        if fixed_positions:
            for i, name in enumerate(['left_team', 'left_team_direction',
                                      'right_team', 'right_team_direction']):
                o.extend(do_flatten(obs[name]))
                # If there were less than 11vs11 players we backfill missing values
                # with -1.
                if len(o) < (i + 1) * 22:
                    o.extend([-1] * ((i + 1) * 22 - len(o)))
        else:
            o.extend(do_flatten(obs['left_team']))
            o.extend(do_flatten(obs['left_team_direction']))
            o.extend(do_flatten(obs['right_team']))
            o.extend(do_flatten(obs['right_team_direction']))

        # If there were less than 11vs11 players we backfill missing values with
        # -1.
        # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
        if len(o) < 88:
            o.extend([-1] * (88 - len(o)))

        # ball position
        o.extend(obs['ball'])
        # ball direction
        o.extend(obs['ball_direction'])
        # one hot encoding of which team owns the ball
        if obs['ball_owned_team'] == -1:
            o.extend([1, 0, 0])
        if obs['ball_owned_team'] == 0:
            o.extend([0, 1, 0])
        if obs['ball_owned_team'] == 1:
            o.extend([0, 0, 1])

        active = [0] * 11
        if obs['active'] != -1:
            active[obs['active']] = 1
        o.extend(active)

        game_mode = [0] * 7
        game_mode[obs['game_mode'].value] = 1
        o.extend(game_mode)
        final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)


def is_goalkeeper(obs):
    player_roles = obs['left_team_roles']
    active = obs['active']
    return player_roles[active] == PlayerRole.GoalKeeper


def is_9(obs):
    player_roles = obs['left_team_roles']
    return player_roles[obs['active']] == PlayerRole.CentralFront


def is_mid_up_front(obs, controlled_player_pos):
    return abs(controlled_player_pos[0]) < SECTOR_SIZE and abs(controlled_player_pos[1]) < SECTOR_SIZE


def is_defender_or_gk(obs):
    player_roles = obs['left_team_roles']
    role = player_roles[obs['active']]
    return role in [PlayerRole.CenterBack,
                    PlayerRole.DefenceMidfield,
                    PlayerRole.LeftBack,
                    PlayerRole.RightBack,
                    PlayerRole.GoalKeeper]


def is_defender(obs):
    player_roles = obs['left_team_roles']
    role = player_roles[obs['active']]
    return role in [PlayerRole.CenterBack,
                    PlayerRole.DefenceMidfield,
                    PlayerRole.LeftBack,
                    PlayerRole.RightBack]


def is_attacker(obs):
    player_roles = obs['left_team_roles']
    role = player_roles[obs['active']]
    return role in [PlayerRole.CentralFront,
                    PlayerRole.LeftMidfield,
                    PlayerRole.RIghtMidfield]


def offside_line(obs):
    line = -1
    is_keeper = True
    for player in obs['right_team']:
        if is_keeper:
            is_keeper = False
            continue

        player_x = player[0]
        if player_x > line:
            line = player_x
    return line


def defense_line(obs):
    line = 1
    is_keeper = True
    for player in obs['left_team']:
        if is_keeper:
            is_keeper = False
            continue

        player_x = player[0]
        if player_x < line:
            line = player_x
    return line


def is_defender_behind(obs, controlled_player_pos):
    controlled_distance_to_goal = distance(controlled_player_pos, [-1, 0])
    for player in obs['left_team']:
        distance_to_goal = distance(player, [-1, 0])
        if distance_to_goal < controlled_distance_to_goal - 0.2:
            return True
    return False


def are_defenders_behind(obs, controlled_player_pos):
    defs = 0
    controlled_distance_to_goal = distance(controlled_player_pos, [-1, 0])
    for player in obs['left_team']:
        distance_to_goal = distance(player, [-1, 0])
        if distance_to_goal < controlled_distance_to_goal - SAFE_DISTANCE:
            defs += 1
    return defs >= 3


def is_defender_blocking_pass(obs, controlled_player_pos, ball_2d_pos, ball_2d_dir):
    controlled_player_pos_x = controlled_player_pos[0]

    for player in obs['right_team']:
        ball_to_player_dir = direction(player, ball_2d_pos)
        if controlled_player_pos_x - SAFE_DISTANCE < player[0] < controlled_player_pos_x + SAFE_DISTANCE:
            if direction_diff_deg(ball_2d_dir, ball_to_player_dir) <= 5:
                return True
    return False


def get_goalkeeper_distance(obs, controlled_player_pos):
    goalie = obs['right_team'][PlayerRole.GoalKeeper.value]
    return distance(goalie, controlled_player_pos)


def dir_distance(_dir):
    _dir_x = _dir[0]
    _dir_y = _dir[1]
    return math.sqrt(_dir_x * _dir_x + _dir_y * _dir_y)


def distance(pos1, pos2):
    _dir = direction(pos1, pos2)
    _dir_x = _dir[0]
    _dir_y = _dir[1]
    return math.sqrt(_dir_x * _dir_x + _dir_y * _dir_y)


def direction(dir_to, dir_from):
    return [dir_to[0] - dir_from[0],
            dir_to[1] - dir_from[1]]


def opposite_dir(_dir):
    return [x * (-1) for x in _dir]


def direction_mul(_dir, factor):
    return [x * factor for x in _dir]


def add(vec1, vec2):
    return [vec1[0] + vec2[0], vec1[1] + vec2[1]]


def direction_diff_deg(dir1, dir2):
    dir1 = np.array(dir1)
    dir2 = np.array(dir2)
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)
    if norm1 == 0 or norm2 == 0:
        return 0

    unit_vector_1 = dir1 / norm1
    unit_vector_2 = dir2 / norm2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.rad2deg(np.arccos(dot_product))


def get_future_pos(current_pos, current_dir, steps=1):
    future_pos = [current_pos[0] + current_dir[0] * steps,
                  current_pos[1] + current_dir[1] * steps]  # TODO: verify if dir is speed (change during step)
    return future_pos


def get_closest_running_dir(running_dir):
    dir_x = running_dir[0]
    dir_y = running_dir[1]

    if dir_y == 0:
        if dir_x >= 0:
            return Action.Right
        else:
            return Action.Left

    rate = abs(dir_x / dir_y)

    if dir_x >= 0:
        if dir_y >= 0:
            # bottom-right
            if rate < 1 / 3:
                return Action.Bottom
            elif rate < 2 / 3:
                return Action.BottomRight
            else:
                return Action.Right
        else:
            # top-right
            if rate < 1 / 3:
                return Action.Top
            elif rate < 2 / 3:
                return Action.TopRight
            else:
                return Action.Right
    else:
        if dir_y >= 0:
            # bottom-left
            if rate < 1 / 3:
                return Action.Bottom
            elif rate < 2 / 3:
                return Action.BottomLeft
            else:
                return Action.Left
        else:
            # top-left
            if rate < 1 / 3:
                return Action.Top
            elif rate < 2 / 3:
                return Action.TopLeft
            else:
                return Action.Left


def action_to_dir(action):
    if action == Action.Top:
        return [0, -1]
    elif action == Action.TopRight:
        return [1, -1]
    elif action == Action.Right:
        return [1, 0]
    elif action == Action.BottomRight:
        return [1, 1]
    elif action == Action.Bottom:
        return [0, 1]
    elif action == Action.BottomLeft:
        return [-1, 1]
    elif action == Action.Left:
        return [-1, 0]
    elif action == Action.TopLeft:
        return [-1, -1]


def is_offside(obs, player_pos_x):
    if player_pos_x <= HALF:
        return False
    players_in_front = 0

    for player in obs['right_team']:
        player_x = player[0]

        if player_x > player_pos_x:
            players_in_front += 1

    return players_in_front <= 1


def is_1_on_1(obs, controlled_player_pos):
    controlled_player_pos_x = controlled_player_pos[0]
    marking_defs = []
    players_in_front = 0

    for player in obs['right_team']:
        player_x = player[0]
        player_y = player[1]
        def_distance = distance(player, controlled_player_pos)
        if def_distance < SAFE_DISTANCE:
            marking_defs.append(player)

        if abs(player_y - controlled_player_pos[1]) < SECTOR_SIZE and player_x > controlled_player_pos_x + SAFE_DISTANCE:
            players_in_front += 1

    return players_in_front <= 1, marking_defs


def is_friend_in_front(obs, controlled_player_pos):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    for player in obs['left_team']:
        player_x = player[0]
        player_y = player[1]

        if abs(player_y - controlled_player_pos_y) < SAFE_DISTANCE:
            if player_x > controlled_player_pos_x + SAFE_DISTANCE:
                return True
    return False


def is_friend_on_wing_back(obs, controlled_player_pos):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    for player in obs['left_team']:
        player_x = player[0]
        player_y = player[1]

        if abs(player_y - controlled_player_pos_y) < SAFE_DISTANCE:
            if player_x < controlled_player_pos_x - SAFE_DISTANCE:
                return True
    return False


def is_free_player_in_front(obs, controlled_player_pos, controlled_player_dir, preferred_side=None):
    active = obs['active']

    front_of_player = get_future_pos(controlled_player_pos, controlled_player_dir, steps=5)
    front_of_player_x = front_of_player[0]
    front_of_player_y = front_of_player[1]

    for player in obs['right_team']:
        player_x = player[0]
        player_y = player[1]

        if abs(player_x - front_of_player_x) < SECTOR_SIZE and \
                abs(player_y - front_of_player_y) < SECTOR_SIZE:
            return False

    front_of_player = get_future_pos(controlled_player_pos, controlled_player_dir, steps=8)
    front_of_player_x = front_of_player[0]
    front_of_player_y = front_of_player[1]

    k = 0
    for player in obs['left_team']:
        if k == active:
            continue
        player_x = player[0]
        player_y = player[1]

        if abs(player_x - front_of_player_x) < SECTOR_SIZE and \
                abs(player_y - front_of_player_y) < SECTOR_SIZE:
            return True


def is_dangerous(obs, controlled_player_pos, direction):
    front_of_player = get_future_pos(controlled_player_pos, direction, steps=1)
    # far_front_of_player = get_future_pos(controlled_player_pos, direction, steps=3)

    for player in obs['right_team']:
        if distance(front_of_player, player) < SAFE_DISTANCE:
            return True

    return False


def is_opp_in_area(obs, position):
    for player in obs['right_team']:
        if distance(position, player) < SAFE_DISTANCE:
            return True

    return False


def is_friend_in_area(obs, position):
    for player in obs['left_team']:
        if distance(position, player) < SAFE_DISTANCE:
            return True

    return False


def friends_in_area(obs, position):
    friends = 0
    for player in obs['left_team']:
        if distance(position, player) < SAFE_DISTANCE:
            friends += 1

    return friends


def is_opp_in_sector(obs, position):
    for player in obs['right_team']:
        if distance(position, player) < SECTOR_SIZE:
            return True

    return False


def is_friend_in_sector(obs, position):
    for player in obs['left_team']:
        if distance(position, player) < SECTOR_SIZE:
            return True

    return False


# DIRECTIONAL

def smart_dir(_dir):
    dir_x = _dir[0]
    if dir_x >= 0:
        return [dir_x / 2, _dir[1]]
    else:
        return [2 * dir_x, _dir[1]]


def walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir):
    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    ball_2d_dir = [ball_dir[0], ball_dir[1]]
    ball_speed = dir_distance(ball_2d_dir)
    if ball_speed > SPRINTING_SPEED:
        future_ball_pos = get_future_pos(ball_2d_pos, ball_2d_dir, steps=6)
    else:
        future_ball_pos = get_future_pos(ball_2d_pos, ball_2d_dir, steps=1)
    direction_to_ball = direction(future_ball_pos, controlled_player_pos)
    running_dir = get_closest_running_dir(direction_to_ball)
    if ball_2d_pos[0] < -1 + SAFE_DISTANCE / 2 and abs(controlled_player_pos[1]) <= POST and running_dir == Action.Left:
        return Action.ReleaseDirection
    return running_dir


def sprint_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir):
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    d = distance(ball_2d_pos, controlled_player_pos)
    if d > 0.5:
        steps = 8
    elif d > SECTOR_SIZE:
        steps = 4
    else:
        steps = 2

    future_ball_pos = get_future_pos(ball_2d_pos, ball_dir, steps=steps)
    direction_to_future_ball = smart_dir(direction(future_ball_pos, controlled_player_pos))
    running_dir = get_closest_running_dir(direction_to_future_ball)
    if ball_2d_pos[0] < -1 + SAFE_DISTANCE / 2 and abs(controlled_player_pos[1]) <= POST and running_dir == Action.Left:
        return Action.ReleaseDirection
    return running_dir


def predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir):
    def fallback():
        future_ball_pos = get_future_pos(ball_2d_pos, ball_dir, steps=1)
        direction_to_future_ball = smart_dir(direction(future_ball_pos, controlled_player_pos))
        running_dir = get_closest_running_dir(direction_to_future_ball)
        return running_dir

    to_ball_dir = direction(ball_2d_pos, controlled_player_pos)
    d = dir_distance(to_ball_dir)
    ball_speed = dir_distance(ball_dir)
    if ball_speed < PASS_SPEED or d < SPRINTING_SPEED:
        return fallback()

    player_speed = dir_distance(controlled_player_dir)
    p_x = controlled_player_pos[0]
    p_y = controlled_player_pos[1]

    steps = d / ball_speed
    player_range = max([player_speed, RUNNING_SPEED]) * steps

    player_trajectory_1 = lambda x: math.sqrt(player_range * player_range - (x - p_x) * (x - p_x)) + p_y if \
        abs(x - p_x) < player_range else -1
    player_trajectory_2 = lambda x: p_y - math.sqrt(player_range * player_range - (x - p_x) * (x - p_x)) if \
        abs(x - p_x) < player_range else -1

    ball_pos_x = ball_2d_pos[0]
    ball_dir_x = ball_dir[0]
    ball_dir_y = ball_dir[1]
    if ball_dir_x != 0:
        ball_dir_const = ball_dir_y / ball_dir_x
        ball_trajectory_const = ball_2d_pos[1] - ball_2d_pos[0] * ball_dir_const
        ball_trajectory = lambda x: x * ball_dir_const + ball_trajectory_const

        solutions = solve_eq(ball_trajectory, player_trajectory_1, player_trajectory_2)
    else:
        sol1_y = player_trajectory_1(ball_pos_x)
        sol2_y = player_trajectory_2(ball_pos_x)
        solutions = [
            [ball_pos_x, sol1_y],
            [ball_pos_x, sol2_y]
        ]

    if not solutions:
        return fallback()

    best_x = -1
    best_y = -1
    if d < player_range:  # ball is inside the cirle of player range
        if ball_dir[0] >= 0:  # take solution further to RIGHT
            for x, y in solutions:
                if x > ball_2d_pos[0]:
                    best_x = x
                    best_y = y
        else:  # take solution further to LEFT
            for x, y in solutions:
                if x < ball_2d_pos[0]:
                    best_x = x
                    best_y = y
    else:  # ball is outside, going from far
        for x, y in solutions:  # take solution further to RIGHT
            if x > best_x:
                best_x = x
                best_y = y

    if best_x == -1:
        return fallback()

    first_touch_point = [best_x, best_y]
    direction_to_future_ball = direction(first_touch_point, controlled_player_pos)
    return get_closest_running_dir(direction_to_future_ball)


def receive_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]
    ball_dir_x = ball_dir[0]
    ball_dir_y = ball_dir[1]

    if ball_dir_x == 0:
        if abs(controlled_player_pos_y - ball_dir_y) < 0.01:
            return Action.ReleaseDirection
        else:
            return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

    ball_dir_const = ball_dir_y / ball_dir_x
    ball_trajectory_const = ball_2d_pos[1] - ball_2d_pos[0] * ball_dir_const
    ball_trajectory = lambda x: x * ball_dir_const + ball_trajectory_const
    if abs(controlled_player_pos_y - ball_trajectory(controlled_player_pos_x)) < 0.01:
        return Action.ReleaseDirection
    else:
        return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)


def rush_keeper(obs, controlled_player_pos, ball_2d_pos, ball_dir):
    d = distance(ball_2d_pos, controlled_player_pos)
    if d < SAFE_DISTANCE:
        return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    if d > 0.5:
        steps = 9
    elif d > SECTOR_SIZE:
        steps = 6
    else:
        steps = 3

    future_ball_pos = get_future_pos(ball_2d_pos, ball_dir, steps=steps)
    direction_to_future_ball = direction(future_ball_pos, controlled_player_pos)
    return get_closest_running_dir(direction_to_future_ball)


# ATTACK

def center_back_play(obs, controlled_player_pos, running_dir, modeled_action):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    # mod_obs = custom_convert_observation([obs], None)
    # action, _states = transfer_ball_up_field_model.predict(mod_obs)
    # if action is not None:
    #     return Action(int(action) + 1)

    # if modeled_action is not None:
    #     try:
    #         return Action(int(modeled_action))
    #     except:
    #         pass

    if running_dir == Action.Left or Action.Left in obs['sticky_actions']:  # never pass straight back left
        return protect_ball(obs, controlled_player_pos, running_dir)

    if is_opp_in_area(obs, controlled_player_pos):
        if controlled_player_pos_x < -LAST_THIRD:
            if controlled_player_pos_y > 0 and Action.Top not in obs['sticky_actions']:
                return Action.ShortPass
            elif controlled_player_pos_y > 0 and Action.Top not in obs['sticky_actions']:
                return Action.ShortPass

    return pass_to_wingers(obs, controlled_player_pos, running_dir)


def pass_to_wingers(obs, controlled_player_pos, running_dir):
    if running_dir in [Action.TopRight, Action.BottomRight] and \
            any([action in obs['sticky_actions'] for action in [Action.TopRight, Action.BottomRight]]):
        return Action.LongPass
    elif running_dir in [Action.Top, Action.Bottom] and \
            any([action in obs['sticky_actions'] for action in [Action.Top, Action.Bottom]]):
        if is_opp_in_sector(obs, controlled_player_pos):
            pass_dir = action_to_dir(running_dir)
            if not is_opp_in_sector(obs, add(controlled_player_pos, pass_dir)):
                if is_friend_in_sector(obs, add(controlled_player_pos, pass_dir)):
                    return Action.ShortPass
        return protect_ball(obs, controlled_player_pos, running_dir)
    elif controlled_player_pos[1] > 0:
        if not is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, SAFE_DISTANCE]):
            return Action.BottomRight
        elif not is_dangerous(obs, controlled_player_pos, [0, SAFE_DISTANCE]):
            return Action.Bottom
        else:
            return protect_ball(obs, controlled_player_pos, running_dir)
    else:
        if not is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, -SAFE_DISTANCE]):
            return Action.TopRight
        elif not is_dangerous(obs, controlled_player_pos, [0, -SAFE_DISTANCE]):
            return Action.Top
        else:
            return protect_ball(obs, controlled_player_pos, running_dir)


def midfield_play(obs, controlled_player_pos, running_dir, modeled_action):
    if running_dir == Action.Left or Action.Left in obs['sticky_actions']:  # never pass straight back left
        return protect_ball(obs, controlled_player_pos, running_dir)

    roles = obs['left_team_roles']
    index = roles.index(PlayerRole.CentralFront)
    striker = obs['left_team'][index]
    is_striker_marked = is_opp_in_sector(obs, striker)
    if not is_offside(obs, striker[0]) and not is_striker_marked:
        if running_dir == Action.Right and Action.Right in obs['sticky_actions']:
            return Action.LongPass
        elif not is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
            return Action.Right
        else:
            return protect_ball(obs, controlled_player_pos, running_dir)

    if offside_line(obs) < SECTOR_SIZE and is_friend_in_front(obs, controlled_player_pos):
        if running_dir in [Action.Right, Action.TopRight, Action.BottomRight] and \
                any([action in obs['sticky_actions'] for action in [Action.Right, Action.TopRight, Action.BottomRight]]):
            return Action.LongPass

    # mod_obs = custom_convert_observation([obs], None)
    # action, _states = transfer_ball_up_field_model.predict(mod_obs)
    # if action is not None:
    #     return Action(int(action) + 1)

    # if modeled_action is not None:
    #     try:
    #         return Action(int(modeled_action))
    #     except:
    #         pass

    return pass_to_wingers(obs, controlled_player_pos, running_dir)


def wing_play(obs, controlled_player_pos, running_dir, ball_pos, ball_dir, modeled_action):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    if controlled_player_pos_x > LAST_THIRD:
        return wing_run(obs, controlled_player_pos, running_dir, ball_pos, ball_dir, modeled_action)
    elif running_dir in [Action.Right, Action.TopRight, Action.BottomRight]:
        if is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
            if is_friend_in_front(obs, controlled_player_pos):
                if abs(controlled_player_pos_y) < 0.4:
                    if controlled_player_pos_y > 0 and Action.BottomRight not in obs['sticky_actions']:
                        return Action.BottomRight
                    elif controlled_player_pos_y < 0 and Action.TopRight not in obs['sticky_actions']:
                        return Action.TopRight
                return Action.LongPass
            else:
                return dribble_into_empty_space(obs, controlled_player_pos, running_dir)
        else:
            if controlled_player_pos_x < HALF and offside_line(obs) > LAST_THIRD:
                return dribble_into_empty_space(obs, controlled_player_pos, running_dir)
            else:
                return wing_run(obs, controlled_player_pos, running_dir, ball_pos, ball_dir, modeled_action)
    elif running_dir == Action.Left and Action.Left in obs['sticky_actions']:
        if is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
            if is_friend_on_wing_back(obs, controlled_player_pos):
                return Action.ShortPass

            if Action.Dribble not in obs['sticky_actions']:
                return Action.Dribble

            if controlled_player_pos_x >= -1 + SHOOTING_DISTANCE:
                if Action.Sprint not in obs['sticky_actions']:
                    return Action.Sprint
                return Action.Left

            if controlled_player_pos_x < -1 + SHOOTING_DISTANCE:
                if Action.Sprint in obs['sticky_actions']:
                    return Action.ReleaseSprint
                return Action.ShortPass
        else:
            return Action.Right
    else:
        return dribble_into_empty_space(obs, controlled_player_pos, running_dir)


def wing_run(obs, controlled_player_pos, running_dir, ball_pos, ball_dir, modeled_action):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]
    ball_pos_y = ball_pos[1]

    if controlled_player_pos_x > 1 - BRAKING_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

        # mod_obs = custom_convert_observation([obs], None)
        # action, _states = assist_cross_model.predict(mod_obs)
        # if action is not None:
        #     return Action(int(action) + 1)

        # if modeled_action is not None:
        #     try:
        #         return Action(int(modeled_action))
        #     except:
        #         pass

        if is_opp_in_area(obs, controlled_player_pos):
            goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
            goal_dir_action = get_closest_running_dir(goal_dir)
            if running_dir in [goal_dir_action, Action.Top, Action.Bottom]:
                return cross(obs, running_dir)
            else:
                return goal_dir_action

        if running_dir == Action.Right:  # in obs['sticky_actions']:
            if controlled_player_pos_y > HALF:
                return Action.Top
            else:
                return Action.Bottom

        if Action.Dribble not in obs['sticky_actions']:
            return Action.Dribble
        if controlled_player_pos_y > HALF:
            return Action.Top
        else:
            return Action.Bottom

    # when ball will go out of play when keep sprinting
    elif (ball_pos_y >= 0.38 and ball_dir[1] > 0) or (ball_pos_y <= -0.38 and ball_dir[1] < 0):
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

    elif controlled_player_pos_x > 1 - 2 * BRAKING_DISTANCE:  # and not is_opp_in_area(obs, controlled_player_pos):
        if any([action in obs['sticky_actions'] for action in [Action.Right, Action.TopRight, Action.BottomRight]]):
            if Action.Sprint not in obs['sticky_actions']:
                return Action.Sprint
        if controlled_player_pos_y > HALF:
            return Action.TopRight
        else:
            return Action.BottomRight

    else:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint


def cross(obs, running_dir):
    for player in obs['left_team']:
        if player[0] > 1 - SHOOTING_DISTANCE and abs(player[1]) < SECTOR_SIZE / 2:
            return Action.ShortPass
    # is_9_ready = abs(obs['left_team'][PlayerRole.CentralFront.value][1]) <= SAFE_DISTANCE
    # if is_9_ready:
    #     return Action.ShortPass
    return running_dir


def beat_goalkeeper(obs, controlled_player_pos):
    controlled_player_pos_y = controlled_player_pos[1]
    striker_y_to_x = abs(controlled_player_pos_y / controlled_player_pos[0])

    roles = obs['right_team_roles']
    index = roles.index(PlayerRole.GoalKeeper)
    goalie = obs['right_team'][index]
    goalie_y = goalie[1]
    goal_to_keeper_axis = direction(goalie, [1, 0])
    goalie_y_to_x = abs(goal_to_keeper_axis[1] / goal_to_keeper_axis[0])

    goal_dir = direction([1, 0], controlled_player_pos)
    distance_to_keeper = distance(controlled_player_pos, goalie)
    if distance_to_keeper > 0.5:  # TODO: set logical distance
        return get_closest_running_dir(goal_dir)

    if goalie_y >= 0:
        if controlled_player_pos_y <= 0:
            return Action.TopRight
        else:
            if striker_y_to_x > goalie_y_to_x:
                if striker_y_to_x > 0.75:
                    return Action.Right
                else:
                    return Action.BottomRight
            else:
                return Action.TopRight
    else:
        if controlled_player_pos_y >= 0:
            return Action.BottomRight
        else:
            if striker_y_to_x > goalie_y_to_x:
                if striker_y_to_x > 0.75:
                    return Action.Right
                else:
                    return Action.TopRight
            else:
                return Action.BottomRight


def protect_ball(obs, controlled_player_pos, running_dir):
    controlled_player_pos_y = controlled_player_pos[1]

    if controlled_player_pos_y < -SAFE_DISTANCE and running_dir == Action.TopLeft:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
    elif controlled_player_pos_y > SAFE_DISTANCE and running_dir == Action.BottomLeft:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

    elif Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint
    if Action.Dribble not in obs['sticky_actions']:
        return Action.Dribble

    elif controlled_player_pos_y > HALF:
        if is_dangerous(obs,controlled_player_pos, [SAFE_DISTANCE, SAFE_DISTANCE]):
            return Action.Bottom
        else:
            return Action.BottomRight
    else:
        if is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, -SAFE_DISTANCE]):
            return Action.Top
        else:
            return Action.TopRight


def lash(obs, controlled_player_pos, goal_dir):
    if is_opp_in_area(obs, controlled_player_pos):
        if goal_dir not in obs['sticky_actions']:
            return goal_dir
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
        return goal_dir
    else:
        return Action.LongPass


def cruise(obs, current_running_action):
    if Action.Dribble not in obs['sticky_actions']:
        return Action.Dribble

    return current_running_action


def finalize_action(obs, controlled_player_pos, goal_dir_action, modeled_action):
    if controlled_player_pos[0] > 1 - BRAKING_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

        # mod_obs = custom_convert_observation([obs], None)
        # action, _states = assist_cross_model.predict(mod_obs)
        # if action is not None:
        #     return Action(int(action) + 1)

        # if modeled_action is not None:
        #     try:
        #         return Action(int(modeled_action))
        #     except:
        #         pass

    elif controlled_player_pos[0] < 1 - BRAKING_DISTANCE:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
        if not is_opp_in_area(obs, controlled_player_pos):
            return beat_goalkeeper(obs, controlled_player_pos)
    return goal_dir_action


def dribble_into_empty_space(obs, controlled_player_pos, running_dir):
    controlled_player_pos_y = controlled_player_pos[1]
    if abs(controlled_player_pos_y) > WING:
        return Action.Right

    if Action.Dribble not in obs['sticky_actions']:
        return Action.Dribble

    preferred_action = Action.TopRight if controlled_player_pos_y <= 0 else Action.BottomRight
    worse_action = Action.TopRight if controlled_player_pos_y > HALF else Action.BottomRight

    if not is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
        return Action.Right
    elif not is_dangerous(obs, controlled_player_pos, direction_mul(action_to_dir(preferred_action), SAFE_DISTANCE)):
        return preferred_action
    elif not is_dangerous(obs, controlled_player_pos, direction_mul(action_to_dir(worse_action), SAFE_DISTANCE)):
        return worse_action
    else:
        return protect_ball(obs, controlled_player_pos, running_dir)


def play_9(obs, controlled_player_pos, running_dir, marking_defs, goal_dir_action, modeled_action):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    # def_line_x = offside_line(obs)

    # marking defs can be challenged
    if marking_defs:
        if all([marking_def[0] < controlled_player_pos_x + SPRINTING_SPEED for marking_def in marking_defs]):
            return finalize_action(obs, controlled_player_pos, goal_dir_action, modeled_action)

    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    # marking defs position
    marking_defs_y = 0
    for marking_def in marking_defs:
        marking_defs_y += marking_def[1]

    roles = obs['left_team_roles']
    lw_index = roles.index(PlayerRole.LeftMidfield)
    left_winger = obs['left_team'][lw_index]
    rw_index = roles.index(PlayerRole.LeftMidfield)
    right_winger = obs['left_team'][rw_index]

    if not marking_defs:  # pass to winger
        if running_dir in [Action.Top, Action.TopRight, Action.Right, Action.BottomRight, Action.Bottom]:
            wing_pass = long_pass_to_wingers(obs, left_winger, right_winger, running_dir)
            if wing_pass is not None:
                return wing_pass

        if Action.Dribble not in obs['sticky_actions']:
            return Action.Dribble

        # turn towards higher positioned winger who is not offside
        if left_winger[0] > right_winger[0] and not is_offside(obs, left_winger[0]):
            return Action.TopRight
        elif not is_offside(obs, right_winger[0]):
            return Action.BottomRight
        else:
            return dribble_into_empty_space(obs, controlled_player_pos, running_dir)
    else:
        if running_dir in [Action.Left, Action.BottomLeft, Action.TopLeft]:
            # turn opposite side to marking defs
            if marking_defs_y / len(marking_defs) >= controlled_player_pos_y:
                return Action.Top
            else:
                return Action.Bottom
        else:
            wing_pass = long_pass_to_wingers(obs, left_winger, right_winger, running_dir)
            if wing_pass is not None:
                return wing_pass
            else:
                return protect_ball(obs, controlled_player_pos, running_dir)


def long_pass_to_wingers(obs, left_winger, right_winger, running_dir):
    if Action.TopRight in obs['sticky_actions']:
        if not is_offside(obs, left_winger[0]):
            return Action.LongPass
    if Action.BottomRight in obs['sticky_actions']:
        if not is_offside(obs, right_winger[0]):
            return Action.LongPass
    elif running_dir in [Action.Top, Action.TopRight]:
        return Action.TopRight
    elif running_dir in [Action.BottomRight, Action.Bottom]:
        return Action.BottomRight

    return None


def goalkeeper_play(obs, controlled_player_pos, controlled_player_dir, running_dir):
    controlled_player_pos_y = controlled_player_pos[1]

    if abs(controlled_player_pos_y) > WING:
        if Action.Right not in obs['sticky_actions']:
            return Action.Right
        return Action.HighPass

    desired_empty_distance = 0.3
    action_sets = [[Action.Top, Action.Bottom], [Action.TopRight, Action.BottomRight]]
    desired_actions = []
    for actions in action_sets:
        for action in actions:
            direction_norm = action_to_dir(action)
            direction = [x * desired_empty_distance for x in direction_norm]
            if not is_opp_in_sector(obs, add(controlled_player_pos, direction)):
                if is_friend_in_sector(obs, controlled_player_pos + direction):
                    desired_actions.append(action)
        if desired_actions:
            break

    if desired_actions:
        action_dir = get_closest_running_dir(controlled_player_dir)
        if action_dir in desired_actions:
            return Action.ShortPass

        last_action = None
        for action in desired_actions:
            if action in obs['sticky_actions']:
                last_action = action
        return last_action or random.choice(desired_actions)

    return protect_ball(obs, controlled_player_pos, running_dir)


# DEFENSE

def retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir):
    ball_height = ball_pos[2]
    controlled_player_pos_x = controlled_player_pos[0]
    ball_pos_x = ball_pos[0]

    ball_2d_dir = [ball_dir[0], ball_dir[1]]
    ball_2d_pos = [ball_pos[0], ball_pos[1]]

    future_ball_2d_pos = get_future_pos(ball_2d_pos, ball_2d_dir, 2)

    ball_speed = dir_distance(ball_dir)
    if ball_speed > SPRINTING_SPEED:
        direction_to_ball = direction(future_ball_2d_pos, controlled_player_pos)
    else:
        direction_to_ball = direction(ball_2d_pos, controlled_player_pos)

    if SLIDING_TACKLE_MODE and ball_height < 0.13:
        tackle_reasons = [SLIDING_TACKLE_MODE > 0 and friends_in_area(obs, controlled_player_pos) >= 2 and controlled_player_pos_x < ball_pos_x,
                          SLIDING_TACKLE_MODE == 1 and is_attacker(obs) and controlled_player_pos_x < ball_pos_x,
                          SLIDING_TACKLE_MODE == 2 and not is_defender(obs) and controlled_player_pos_x < ball_pos_x]
        if any(tackle_reasons):
            if direction_diff_deg(controlled_player_dir, direction_to_ball) < 45:
                distance_to_ball = dir_distance(direction_to_ball)
                if distance_to_ball <= SLIDING_TACKLE_RANGE:
                    return Action.Slide

    ball_pos_x = ball_pos[0]
    controlled_player_pos_x = controlled_player_pos[0]

    if controlled_player_pos_x >= ball_pos_x:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
    else:  # controlled_player_pos_x < ball_pos_x:  # defender closer to goal than ball
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

    return get_closest_running_dir(direction_to_ball)


def rush_to_defense(obs, controlled_player_pos, ball_pos):
    def_line = defense_line(obs)
    position = [def_line, ball_pos[1] / 2]
    d = distance(position, controlled_player_pos)

    if d < SAFE_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
    else:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

    direction_to_pos = direction(position, controlled_player_pos)
    return get_closest_running_dir(direction_to_pos)


def rush_to_own_goal(obs, controlled_player_pos, running_dir):
    goal_dir = [-1 - controlled_player_pos[0], -controlled_player_pos[1]]
    goal_dir_action = get_closest_running_dir(goal_dir)
    if running_dir != goal_dir_action:
        return goal_dir_action
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    return goal_dir_action


def rush_to_stop_ball(obs, controlled_player_pos, ball_pos, offside_trap=False):
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, steps=1)

    dir_ball_to_goal = direction_mul(direction([-1, 0], future_ball_pos), 1/4)
    position = add(future_ball_pos, dir_ball_to_goal)
    # d = distance(position, controlled_player_pos)

    if offside_trap:
        def_line_x = defense_line(obs)
        position = [
            max(position[0], def_line_x),
            position[1]
        ]

    direction_to_defense = direction(position, controlled_player_pos)
    return get_closest_running_dir(direction_to_defense)


def control_attacker(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir):
    if are_defenders_behind(obs, controlled_player_pos) or ball_pos[0] < controlled_player_pos[0]:
        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    else:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
        ball_distance = distance(ball_pos, controlled_player_pos)
        if ball_distance < SAFE_DISTANCE and not is_opp_in_area(obs, ball_pos):
            to_ball_dir = smart_dir(direction(ball_pos, controlled_player_pos))
            return get_closest_running_dir(to_ball_dir)
        else:
            goal_dir = direction([-1, 0], controlled_player_pos)
            return get_closest_running_dir(goal_dir)


# GOALKEEPER

def stay_front_of_goal(obs, controlled_player_pos, ball_pos):
    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, steps=3)

    dir_ball_to_goal = direction_mul(direction([-1, 0], future_ball_pos), 3/4)
    position = add(future_ball_pos, dir_ball_to_goal)

    d = distance(position, controlled_player_pos)

    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint
    # if d < SAFE_DISTANCE:
    #     if Action.Sprint in obs['sticky_actions']:
    #         return Action.ReleaseSprint
    # else:
    #     if Action.Sprint not in obs['sticky_actions']:
    #         return Action.Sprint

    direction_to_defense = direction(position, controlled_player_pos)
    return get_closest_running_dir(direction_to_defense)


# TRANSITION

def react_to_incoming_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir, ball_to_receiver_dir, running_dir):
    ball_dir_x = ball_dir[0]
    ball_dir_y = ball_dir[1]
    ball_height = ball_pos[2]
    ball_2d_pos = [ball_pos[0], ball_pos[1]]
    ball_2d_dir = [ball_dir_x, ball_dir_y]
    # distance_from_goal = distance(ball_2d_pos, [1, 0])

    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
    # goal_dir_action = get_closest_running_dir(goal_dir)
    #
    # ball_speed = dir_distance(ball_2d_dir)
    # distance_to_ball = distance(ball_pos, controlled_player_pos)

    # if controlled_player_pos_x < ball_pos[0] and distance_to_ball < ball_speed * FIRST_TOUCH_TIMING and ball_height < 0.15:
    #     return first_touch(obs, controlled_player_pos, controlled_player_pos_x, controlled_player_pos_y, running_dir,
    #                        goal_dir_action, distance_from_goal)

    if ball_2d_pos[0] > 1 - SAFE_DISTANCE or controlled_player_pos[0] > ball_2d_pos[0]:
        # TODO: if ball will surpass me then dont release sprint
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
    else:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

    if ball_dir_x > 0 and abs(ball_dir_x) > abs(ball_dir_y):
        if is_defender_blocking_pass(obs, controlled_player_pos, ball_2d_pos, ball_2d_dir):
            return receive_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir)
        else:
            return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    elif ball_dir_x <= 0:
        return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    else:
        return receive_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir)


def first_touch(obs, controlled_player_pos, controlled_player_pos_x, controlled_player_pos_y, running_dir,
                goal_dir_action, distance_from_goal):
    if distance_from_goal <= SHOOTING_DISTANCE and abs(controlled_player_pos_y) < SECTOR_SIZE:
        return Action.Shot
    if controlled_player_pos_x > 1 - SAFE_DISTANCE:
        if controlled_player_pos_y > 0:
            return Action.Top
        else:
            return Action.Bottom

    is_one_on_one = is_1_on_1(obs, controlled_player_pos)

    if is_one_on_one and abs(controlled_player_pos_y) < WING:
        # if Action.Sprint not in obs['sticky_actions']:
        #     return Action.Sprint
        return goal_dir_action
    else:
        if is_opp_in_area(obs, controlled_player_pos):
            if controlled_player_pos_x < -LAST_THIRD:
                return Action.ShortPass
            else:
                if running_dir == Action.Left or Action.Left in obs['sticky_actions']:
                    return random.choice([Action.BottomLeft, Action.TopLeft])
                else:
                    return Action.ShortPass
        else:
            if controlled_player_pos_x > -LAST_THIRD:
                if Action.Sprint not in obs['sticky_actions']:
                    return Action.Sprint
            return dribble_into_empty_space(obs, controlled_player_pos, running_dir)


@human_readable_agent
def agent(obs, modeled_action=None):

    def transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir):
        ball_pos_x = ball_pos[0]
        ball_pos_y = ball_pos[1]
        ball_height = ball_pos[2]
        ball_dir_x = ball_dir[0]
        ball_dir_y = ball_dir[1]
        ball_2d_pos = [ball_pos_x, ball_pos_y]
        ball_2d_dir = [ball_dir_x, ball_dir_y]

        controlled_player_pos_x = controlled_player_pos[0]
        controlled_player_pos_y = controlled_player_pos[1]
        ball_to_receiver_dir = direction(controlled_player_pos, ball_2d_pos)
        ball_speed = dir_distance(ball_2d_dir)
        distance_to_ball = dir_distance(ball_to_receiver_dir)

        # steering goalkeeper
        if is_goalkeeper(obs):
            if ball_height > 0.2:
                return stay_front_of_goal(obs, controlled_player_pos, ball_pos)
            else:
                return receive_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir)

        # is closest to ball
        if ball_speed < PASS_SPEED and distance_to_ball < SAFE_DISTANCE and not is_opp_in_area(obs, ball_2d_pos):
            return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

        # stand in defense
        if ball_dir_x < -0.005 and not are_defenders_behind(obs, controlled_player_pos) and controlled_player_pos_x > LAST_THIRD:
            return rush_to_defense(obs, controlled_player_pos, ball_pos)

        elif ball_dir_x < -0.005 and not are_defenders_behind(obs, controlled_player_pos):
            return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)

        # chase ball on wing
        if abs(controlled_player_pos[1]) > WING and direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) > 30:  # and controlled_player_pos[0] > HALF:
            if controlled_player_pos_x < 1 - BRAKING_DISTANCE or distance_to_ball > 0.025:
                if (0 < controlled_player_pos_y < ball_pos_y or ball_pos_y < controlled_player_pos_y > 0) and abs(ball_to_receiver_dir[0]/ball_to_receiver_dir[1]) < 1.5:
                    return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)
                else:
                    return sprint_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)
            else:
                return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

        # in short distance set shot before first touch
        distance_to_goal = distance([1, 0], ball_2d_pos)
        if distance_to_goal < SHORT_SHOOTING_DISTANCE and abs(controlled_player_pos[1]) < SECTOR_SIZE:
            if ball_speed > PASS_SPEED and direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) <= 5 and ball_height < 0.15:
                return Action.Shot
            elif ball_speed > PASS_SPEED and ball_height < 0.15:
                ball_dir_action = get_closest_running_dir(opposite_dir(ball_to_receiver_dir))
                return random.choice([Action.Shot, ball_dir_action, ball_dir_action, ball_dir_action])
            else:
                return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir)

        # directional first touch
        if direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) <= 30:
            return react_to_incoming_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir,
                                          ball_to_receiver_dir, running_dir)

        # whatever else
        return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir)

    def attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_pos, ball_dir, modeled_action):
        ball_2d_pos = [ball_pos[0], ball_pos[1]]

        # Does the player we control have the ball?
        if int(obs['ball_owned_player']) == int(obs['active']):
            controlled_player_pos_x = controlled_player_pos[0]
            controlled_player_pos_y = controlled_player_pos[1]

            goal_dir = [1 - controlled_player_pos_x, -controlled_player_pos_y]
            goal_dir_action = get_closest_running_dir(goal_dir)
            distance_from_goal = dir_distance(goal_dir)

            # from close range shoot
            if distance_from_goal <= SHORT_SHOOTING_DISTANCE and (controlled_player_pos_y == 0 or abs(controlled_player_pos_x/controlled_player_pos_y) >= 1.25):
                return Action.Shot

            if (distance_from_goal <= SHOOTING_DISTANCE and
                (controlled_player_pos_y == 0 or abs(controlled_player_pos_x/controlled_player_pos_y) >= 1.25)) or \
                    (Action.Sprint in obs['sticky_actions'] and distance_from_goal <= LONG_SHOOTING_DISTANCE and
                     (controlled_player_pos_y == 0 or abs(controlled_player_pos_x/controlled_player_pos_y) >= 2)):
                if goal_dir_action not in obs['sticky_actions']:
                    return goal_dir_action

                if controlled_player_pos_x > 1 - SAFE_DISTANCE:
                    if Action.Sprint in obs['sticky_actions']:
                        return Action.ReleaseSprint
                return Action.Shot

            # keeper out of goal, shoot
            goal_distance = distance(controlled_player_pos, [1, 0])
            keeper_distance = get_goalkeeper_distance(obs, controlled_player_pos)
            if keeper_distance < goal_distance / 2:
                if running_dir != goal_dir_action and Action.Sprint not in obs['sticky_actions']:
                    return goal_dir_action
                return Action.Shot

            # last inches of the field
            if controlled_player_pos_x > 1 - SAFE_DISTANCE:
                if Action.Sprint in obs['sticky_actions']:
                    return Action.ReleaseSprint
                if Action.Dribble not in obs['sticky_actions']:
                    return Action.Dribble

                if controlled_player_pos_y > 0 and \
                        (running_dir != Action.Top or Action.Top not in obs['sticky_actions']):
                    return Action.Top
                elif controlled_player_pos_y < 0 and \
                        (running_dir != Action.Bottom or Action.Bottom not in obs['sticky_actions']):
                    return Action.Bottom

                # mod_obs = custom_convert_observation([obs], None)
                # action, _states = assist_cross_model.predict(mod_obs)
                # if action is not None:
                #     return Action(int(action) + 1)

                # if modeled_action is not None:
                #     try:
                #         return Action(int(modeled_action)), 'trained action'
                #     except:
                #         pass

                if controlled_player_pos_y > 0:
                    if is_dangerous(obs, controlled_player_pos, [0, -SAFE_DISTANCE]):
                        return cross(obs, running_dir)
                    return Action.Top
                elif controlled_player_pos_y < 0:
                    if is_dangerous(obs, controlled_player_pos, [0, SAFE_DISTANCE]):
                        return cross(obs, running_dir)
                    return Action.Bottom
                else:
                    return Action.Shot

            is_one_on_one, marking_defs = is_1_on_1(obs, controlled_player_pos)

            # run straight one on one to beat the keeper
            if is_one_on_one and abs(controlled_player_pos_y) < WING:
                return finalize_action(obs, controlled_player_pos, goal_dir_action, modeled_action)

            # play on the wing
            if abs(controlled_player_pos_y) > WING:
                return wing_play(obs, controlled_player_pos, running_dir, ball_pos, ball_dir, modeled_action)

            # build play from middle of the field when counter attacking
            if controlled_player_pos_x > -LAST_THIRD and (is_9(obs) or is_mid_up_front(obs, controlled_player_pos)) and offside_line(obs) <= LAST_THIRD:
                return play_9(obs, controlled_player_pos, running_dir, marking_defs, goal_dir_action, modeled_action)

            # in the last third try to go straight for goal
            if controlled_player_pos_x > LAST_THIRD:
                return finalize_action(obs, controlled_player_pos, goal_dir_action, modeled_action)

            # other playmaking positions
            if controlled_player_pos_x > -LAST_THIRD:
                return midfield_play(obs, controlled_player_pos, running_dir, modeled_action)

            # kick anywhere but own goal
            if controlled_player_pos_x <= SAFE_DISTANCE and controlled_player_pos_y <= POST:
                if running_dir == Action.Left or Action.Left in obs['sticky_actions']:
                    return Action.Right
                else:
                    return Action.HighPass

            # other defensive playmaking positions
            if controlled_player_pos_x <= -LAST_THIRD:
                if obs['ball_owned_player'] in [PlayerRole.LeftBack, PlayerRole.RightBack]:
                    return protect_ball(obs, controlled_player_pos, running_dir)
                else:
                    return center_back_play(obs, controlled_player_pos, running_dir, modeled_action)

            # should never reach this
            return dribble_into_empty_space(obs, controlled_player_pos, running_dir)
        else:
            return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir):
        controlled_player_pos_x = controlled_player_pos[0]
        ball_pos_x = ball_pos[0]

        if is_goalkeeper(obs):
            return stay_front_of_goal(obs, controlled_player_pos, ball_pos)

        if ball_pos_x > LAST_THIRD and not is_attacker(obs):
            return rush_to_defense(obs, controlled_player_pos, ball_pos)

        elif ball_pos_x > HALF and is_attacker(obs):
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

        elif ball_pos_x > HALF:  # and is_defender(obs):
            if are_defenders_behind(obs, controlled_player_pos):
                return rush_to_stop_ball(obs, controlled_player_pos, ball_pos, offside_trap=True)
            else:
                return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)

        elif -1 + SECTOR_SIZE < ball_pos_x <= controlled_player_pos_x:  # and is_defender(obs):
            return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)

        elif ball_pos_x > -1 + SECTOR_SIZE and ball_pos_x > controlled_player_pos_x:
            if are_defenders_behind(obs, controlled_player_pos):
                return control_attacker(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
            else:
                return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)

        elif ball_pos_x <= -1 + SECTOR_SIZE:
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

    try:
        ball_pos = obs['ball']
        ball_dir = obs['ball_direction']
        game_mode = obs['game_mode']
        ball_2d_pos = [ball_pos[0], ball_pos[1]]
        ball_to_goal_distance = distance(ball_2d_pos, [1, 0])
        controlled_player_pos = obs['left_team'][obs['active']]

        if game_mode != GameMode.Normal:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            if Action.Sprint in obs['sticky_actions']:
                return Action.Dribble

        if game_mode == GameMode.Penalty:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            return Action.Shot

        elif game_mode == GameMode.GoalKick:
            return random.choice([Action.Top, Action.Bottom])

        elif game_mode == GameMode.Corner:
            if controlled_player_pos[1] > 0 and Action.Top not in obs['sticky_actions']:
                return Action.Top
            elif controlled_player_pos[1] < 0 and Action.Bottom not in obs['sticky_actions']:
                return Action.Bottom
            return Action.HighPass

        elif game_mode == GameMode.ThrowIn:
            if ball_pos[0] < LAST_THIRD:
                return random.choice([Action.Right, Action.ShortPass])
            else:
                return Action.LongPass

        elif game_mode == GameMode.FreeKick and ball_to_goal_distance <= FK_SHOOTING_DISTANCE:
            return Action.Shot
        elif game_mode == GameMode.FreeKick and ball_pos[0] >= LAST_THIRD:
            return Action.ShortPass
        elif game_mode == GameMode.FreeKick and ball_pos[0] < LAST_THIRD:
            return random.choice([Action.Top, Action.Bottom, Action.ShortPass])

        elif game_mode == GameMode.KickOff:
            if Action.Sprint not in obs['sticky_actions']:
                return Action.Sprint
            return Action.Right

        elif game_mode != GameMode.Normal:
            return Action.ShortPass

        ball_owned_team = obs['ball_owned_team']
        controlled_player_dir = obs['left_team_direction'][obs['active']]
        running_dir = get_closest_running_dir(controlled_player_dir)

        if ball_owned_team == -1:
            return transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)
        elif ball_owned_team == 0:  # we have the ball
            return attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_2d_pos, ball_dir, modeled_action)
        else:  # ball_owned_team = 1, opponents
            return defend(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    except Exception as e:
        return Action.Idle, 'error'
