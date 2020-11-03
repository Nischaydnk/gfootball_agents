import random
from enum import Enum

import math
import numpy as np
from kaggle_environments.envs.football.helpers import Action, PlayerRole

from sympy import symbols, Eq, solve

WING = 0.32
SAFE_DISTANCE = 0.085
SECTOR_SIZE = 0.2
SHOOTING_DISTANCE = 0.32
SHORT_SHOOTING_DISTANCE = 0.24
FK_SHOOTING_DISTANCE = 0.3
HALF = 0
LAST_THIRD = 0.3
BRAKING_DISTANCE = 0.26
SLIDING_TACKLE_RANGE = 0.06
FIRST_TOUCH_TIMING = 1.5

RUNNING_SPEED = 0.03
DRIBBLING_SPEED = 0.03
SPRINTING_SPEED = 0.05

SLIDING_TACKLE_MODE = 0


class SlidingTackleMode(Enum):
    NEVER = 0
    ONLY_ = 1
    ALWAYS = 2


# LOGGERS

def print_sticky_actions(obs):
    return ','.join([action.name for action in obs['sticky_actions']])


# HELPERS

def is_goalkeeper(obs):
    return int(obs['active']) == int(PlayerRole.GoalKeeper.value)


def is_defender_or_gk(obs):
    return obs['active'] in [PlayerRole.CenterBack,
                             PlayerRole.DefenceMidfield,
                             PlayerRole.LeftBack,
                             PlayerRole.RightBack,
                             PlayerRole.GoalKeeper]


def is_defender(obs):
    return obs['active'] in [PlayerRole.CenterBack,
                             PlayerRole.DefenceMidfield,
                             PlayerRole.LeftBack,
                             PlayerRole.RightBack]


def is_attacker(obs):
    return obs['active'] in [PlayerRole.CentralFront,
                             PlayerRole.LeftMidfield,
                             PlayerRole.RIghtMidfield]


def offside_line(obs):
    line = -1
    is_keeper = True
    for player in obs['right_team']:
        if is_keeper:
            is_keeper = False
            continue

        player_x = player[1]
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

        player_x = player[1]
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
        if distance_to_goal < controlled_distance_to_goal - 0.2:
            defs += 1
    return defs >= 3


def is_defender_blocking_pass(obs, controlled_player_pos, ball_2d_pos, ball_2d_dir):
    controlled_player_pos_x = controlled_player_pos[0]

    for player in obs['right_team']:
        ball_to_player_dir = direction(player, ball_2d_pos)
        if player[0] < controlled_player_pos_x + SAFE_DISTANCE:
            if direction_diff_deg(ball_2d_dir, ball_to_player_dir) <= 15:
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
    future_pos = [current_pos[0] + current_dir[0] * steps, current_pos[1] + current_dir[1] * steps]  # TODO: verify if dir is speed (change during step)
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

        if abs(player_y) - abs(controlled_player_pos[1]) < SECTOR_SIZE and player_x > controlled_player_pos_x:
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

def run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir):
    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    ball_2d_dir = [ball_dir[0], ball_dir[1]]
    ball_speed = dir_distance(ball_2d_dir)
    if ball_speed > SPRINTING_SPEED:
        future_ball_pos = get_future_pos(ball_pos, ball_2d_dir, steps=6)
    else:
        future_ball_pos = get_future_pos(ball_pos, ball_2d_dir, steps=1)
    direction_to_ball = direction(future_ball_pos, controlled_player_pos)
    return get_closest_running_dir(direction_to_ball)


def rush_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir):
    d = distance(ball_2d_pos, controlled_player_pos)
    if d < SAFE_DISTANCE:
        return run_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    if d > 0.5:
        steps = 9
    elif d > SECTOR_SIZE:
        steps = 3
    else:
        steps = 1

    future_ball_pos = get_future_pos(ball_2d_pos, ball_dir, steps=steps)
    direction_to_future_ball = direction(future_ball_pos, controlled_player_pos)
    return get_closest_running_dir(direction_to_future_ball)


def predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir):
    d = distance(ball_2d_pos, controlled_player_pos)
    if d < SAFE_DISTANCE:
        return run_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    ball_speed = dir_distance(ball_dir)
    player_speed = dir_distance(controlled_player_dir)
    p_x = controlled_player_pos[0]
    p_y = controlled_player_pos[1]

    steps = d / ball_speed
    player_range = max([player_speed, RUNNING_SPEED]) * steps

    ball_dir_const = ball_dir[0] / ball_dir[1]
    ball_trajectory_const = ball_2d_pos[1] - ball_2d_pos[0] * ball_dir_const
    ball_trajectory = lambda x: x * ball_dir_const + ball_trajectory_const
    # player_trajectory = lambda x: math.sqrt((x-p_x) * (x-p_x) - player_range * player_range + p_y)

    x = symbols('x')
    eq = Eq(((x-p_x) * (x-p_x) - player_range * player_range + p_y) *
            ((x-p_x) * (x-p_x) - player_range * player_range + p_y) -
            x * ball_dir_const + ball_trajectory_const, 0)
    xs = solve(eq)
    best_x = -1
    for x in xs:
        try:
            real_x = float(x)
            if -1 < real_x < 1 - SAFE_DISTANCE:
                if real_x > best_x:
                    best_x = real_x
        except:
            continue

    if best_x == -1:
        return rush_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)
    best_y = ball_trajectory(best_x)

    with open('./log', mode='a') as f:
        f.write(f'predicting works!!!, {best_x}, {best_y}')
        f.write('\n')
    first_touch_point = [best_x, best_y]
    direction_to_future_ball = direction(first_touch_point, controlled_player_pos)
    return get_closest_running_dir(direction_to_future_ball)


def rush_keeper(obs, controlled_player_pos, ball_2d_pos, ball_dir):
    d = distance(ball_2d_pos, controlled_player_pos)
    if d < SAFE_DISTANCE:
        return run_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

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

def center_back_play(obs, controlled_player_pos, running_dir):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    if controlled_player_pos_y < -SAFE_DISTANCE and running_dir == Action.TopLeft:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
    if controlled_player_pos_y > SAFE_DISTANCE and running_dir == Action.BottomLeft:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    if is_opp_in_area(obs, controlled_player_pos):
        if controlled_player_pos_x < -LAST_THIRD:
            if controlled_player_pos_y > 0 and Action.Top not in obs['sticky_actions']:
                return Action.ShortPass
            elif controlled_player_pos_y > 0 and Action.Top not in obs['sticky_actions']:
                return Action.ShortPass

    if running_dir == Action.Left or Action.Left in obs['sticky_actions']:  # never pass straight back left
        return protect_ball(obs, controlled_player_pos, running_dir)

    # striker = obs['left_team'][PlayerRole.CentralFront.value]
    # is_striker_marked = is_opp_in_sector(obs, striker)
    # if not is_offside(obs, striker[0]) and not is_striker_marked:
    #     if running_dir == Action.Right:
    #         return Action.LongPass
    #     elif not is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
    #         return Action.Right
    #     else:
    #         return protect_ball(obs, controlled_player_pos)

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
            return Action.HighPass
    else:
        if not is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, -SAFE_DISTANCE]):
            return Action.TopRight
        elif not is_dangerous(obs, controlled_player_pos, [0, -SAFE_DISTANCE]):
            return Action.Top
        else:
            return Action.HighPass


def midfield_play(obs, controlled_player_pos, running_dir):
    if running_dir == Action.Left or Action.Left in obs['sticky_actions']:  # never pass straight back left
        return protect_ball(obs, controlled_player_pos, running_dir)

    striker = obs['left_team'][PlayerRole.CentralFront.value]
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

    return pass_to_wingers(obs, controlled_player_pos, running_dir)


def wing_play(obs, controlled_player_pos, running_dir, ball_pos, ball_dir):
    controlled_player_pos_x = controlled_player_pos[0]

    if controlled_player_pos_x > LAST_THIRD:
        return wing_run(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)
    elif running_dir == Action.Right:
        if is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
            if is_friend_in_front(obs, controlled_player_pos):
                return Action.HighPass
            else:
                return dribble_into_empty_space(obs, controlled_player_pos, running_dir)
        else:
            return wing_run(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)
    elif running_dir == Action.Left and Action.Left in obs['sticky_actions']:
        if is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
            if is_friend_on_wing_back(obs, controlled_player_pos):
                return Action.ShortPass

            if Action.Dribble not in obs['sticky_actions']:
                return Action.Dribble

            if controlled_player_pos_x > -LAST_THIRD:
                if Action.Sprint not in obs['sticky_actions']:
                    return Action.Sprint
                return Action.Left

            if controlled_player_pos_x < -LAST_THIRD:
                if Action.Sprint in obs['sticky_actions']:
                    return Action.ReleaseSprint
                return Action.ShortPass
        else:
            return Action.Right
    else:
        return dribble_into_empty_space(obs, controlled_player_pos, running_dir)


def wing_run(obs, controlled_player_pos, running_dir, ball_pos, ball_dir):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    if controlled_player_pos_x > 1 - BRAKING_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

        if is_opp_in_area(obs, controlled_player_pos):
            goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
            goal_dir_action = get_closest_running_dir(goal_dir)
            if running_dir in [goal_dir_action, Action.Top, Action.Bottom]:
                return cross(obs)
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
    elif controlled_player_pos_x > 1 - 2 * BRAKING_DISTANCE and not is_opp_in_area(obs, controlled_player_pos):
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
        if controlled_player_pos_y > HALF:
            return Action.TopRight
        else:
            return Action.BottomRight

    # account for weird situations when winger forgets to take the ball
    if abs(ball_pos[1] - controlled_player_pos_y) > 0.01:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
    if ball_pos[0] < controlled_player_pos_x:
        return rush_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

    if running_dir != Action.Right or Action.Right not in obs['sticky_actions']:  # not in obs['sticky_actions']:
        return Action.Right
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    # TODO: is player in front then do else
    return Action.Right


def cross(obs):
    for player in obs['left_team']:
        if player[0] > 1 - SHOOTING_DISTANCE and abs(player[1]) < SAFE_DISTANCE:
            return Action.ShortPass
    # is_9_ready = abs(obs['left_team'][PlayerRole.CentralFront.value][1]) <= SAFE_DISTANCE
    # if is_9_ready:
    #     return Action.ShortPass
    return Action.HighPass


def beat_goalkeeper(obs, controlled_player_pos):
    controlled_player_pos_y = controlled_player_pos[1]
    striker_y_to_x = abs(controlled_player_pos_y / controlled_player_pos[0])

    goalie = obs['right_team'][PlayerRole.GoalKeeper.value]
    goalie_y = goalie[1]
    goal_to_keeper_axis = direction(goalie, [1, 0])
    goalie_y_to_x = abs(goal_to_keeper_axis[1] / goal_to_keeper_axis[0])

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


def finalize_action(obs, controlled_player_pos, goal_dir_action):
    if controlled_player_pos[0] > 1 - BRAKING_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
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
        # with open('./log', mode='a') as f:
        #     f.write(f'protecting ball, {preferred_action}, {direction_mul(action_to_dir(Action.Right), RUNNING_SPEED)}')
        #     f.write('\n')
        return protect_ball(obs, controlled_player_pos, running_dir)


def play_9(obs, controlled_player_pos, running_dir, marking_defs, goal_dir_action):
    # TODO: WHAT HAPPENS HERE
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    if all([marking_def[0] < controlled_player_pos_x + SPRINTING_SPEED for marking_def in marking_defs]):
        return finalize_action(obs, controlled_player_pos, goal_dir_action)

    if any([action in obs['sticky_actions'] for action in [Action.TopRight, Action.BottomRight]]) and running_dir in [Action.Top, Action.TopRight, Action.BottomRight, Action.Bottom]:
    #if running_dir in [Action.Top, Action.TopRight, Action.BottomRight, Action.Bottom]:
        return Action.LongPass

    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint
    if Action.Dribble not in obs['sticky_actions']:
        return Action.Dribble

    # play long ball to the winger
    marking_defs_y = 0
    for marking_def in marking_defs:
        marking_defs_y += marking_def[1]

    if not marking_defs:  # pass to higher winger
        left_winger = obs['left_team'][PlayerRole.LeftMidfield.value]
        right_winger = obs['left_team'][PlayerRole.RIghtMidfield.value]
        if left_winger[0] > right_winger[0] and not is_offside(obs, left_winger[0]):
            return Action.TopRight
        elif not is_offside(obs, right_winger[0]):
            return Action.BottomRight
        else:
            return protect_ball(obs, controlled_player_pos, running_dir)
    elif marking_defs_y / len(marking_defs) >= controlled_player_pos_y:
        # if running_dir == Action.Top:
        #     return Action.TopRight
        # else:
        return Action.Top
    else:
        # if running_dir == Action.Bottom:
        #     return Action.BottomRight
        # else:
        return Action.Bottom


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

    # with open('./log', mode='a') as f:
    #     f.write(f'keeper options: {",".join([str(action.value) for action in desired_actions])}')
    #     f.write('\n')

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

    if controlled_player_pos_x > ball_pos_x:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
    elif controlled_player_pos_x < ball_pos_x - 0.15:  # defender closer to goal than ball
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
    else:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

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


def rush_to_stop_ball(obs, controlled_player_pos, ball_pos):
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, steps=3)

    dir_ball_to_goal = direction_mul(direction([-1, 0], future_ball_pos), 1/4)
    position = add(future_ball_pos, dir_ball_to_goal)

    d = distance(position, controlled_player_pos)

    if d < SAFE_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
    else:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

    direction_to_defense = direction(position, controlled_player_pos)
    return get_closest_running_dir(direction_to_defense)


# GOALKEEPER

def stay_front_of_goal(obs, controlled_player_pos, ball_pos):
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, steps=3)

    dir_ball_to_goal = direction_mul(direction([-1, 0], future_ball_pos), 3/4)
    position = add(future_ball_pos, dir_ball_to_goal)

    d = distance(position, controlled_player_pos)

    if d < SAFE_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
    else:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

    direction_to_defense = direction(position, controlled_player_pos)
    return get_closest_running_dir(direction_to_defense)


# TRANSITION

def react_to_incoming_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir, ball_to_receiver_dir, running_dir):
    ball_dir_x = ball_dir[0]
    ball_dir_y = ball_dir[1]
    ball_height = ball_pos[2]
    ball_2d_pos = [ball_pos[0], ball_pos[1]]
    ball_2d_dir = [ball_dir_x, ball_dir_y]
    distance_from_goal = distance(ball_pos, [1, 0])

    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
    goal_dir_action = get_closest_running_dir(goal_dir)

    ball_speed = dir_distance(ball_2d_dir)
    distance_to_ball = distance(ball_pos, controlled_player_pos)
    is_9 = int(obs['active']) == int(PlayerRole.CentralFront.value)

    if distance_to_ball < ball_speed * FIRST_TOUCH_TIMING and ball_height < 0.03:
        return first_touch(obs, controlled_player_pos, controlled_player_pos_x, controlled_player_pos_y, running_dir,
                           goal_dir_action, distance_from_goal, is_9)
    elif ball_dir_x > 0 and abs(ball_dir_x) > abs(ball_dir_y):
        # TODO: WHY THEY DONT RUN WITH THE DEFENDER BLOCKING CLAUSE!!!
        if is_defender_blocking_pass(obs, controlled_player_pos, ball_2d_pos, ball_2d_dir):
            return run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)
        else:
            return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    elif ball_dir_x <= 0:
        return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    else:
        return run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)


def first_touch(obs, controlled_player_pos, controlled_player_pos_x, controlled_player_pos_y, running_dir,
                goal_dir_action, distance_from_goal, is_9):
    if distance_from_goal <= SHOOTING_DISTANCE and abs(controlled_player_pos_y) < SECTOR_SIZE:
        return Action.Shot
    is_one_on_one = is_1_on_1(obs, controlled_player_pos)

    if (is_9 or is_one_on_one) and abs(controlled_player_pos_y) < WING:
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
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
