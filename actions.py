import random

import math
import numpy as np
from kaggle_environments.envs.football.helpers import Action, PlayerRole

WING = 0.32
SAFE_DISTANCE = 0.105
SECTOR_SIZE = 0.2
SHOOTING_DISTANCE = 0.32
HALF = 0
LAST_THIRD = 0.3
BRAKING_DISTANCE = 0.32
SLIDING_TACKLE_RANGE = 0.03

RUNNING_SPEED = 0.03
DRIBBLING_SPEED = 0.03
SPRINTING_SPEED = 0.05


# HELPERS

def is_defender(obs):
    return obs['active'] in [PlayerRole.CenterBack,
                             PlayerRole.LeftBack,
                             PlayerRole.RightBack,
                             PlayerRole.GoalKeeper]


def is_attacker(obs):
    return obs['active'] in [PlayerRole.CentralFront,
                             PlayerRole.LeftMidfield,
                             PlayerRole.RIghtMidfield]


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
    future_pos = current_pos + [current_dir[0] * steps, current_dir[1] * steps]  # TODO: verify if dir is speed (change during step)
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
        # player_y = player[1]
        def_distance = distance(player, controlled_player_pos)
        if def_distance < SAFE_DISTANCE:
            marking_defs.append(player)

        if player_x > controlled_player_pos_x:
            players_in_front += 1

    return players_in_front <= 1, marking_defs


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

def run_toward_ball(obs, controlled_player_pos, ball_pos):
    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    direction_to_ball = direction(ball_pos, controlled_player_pos)
    return get_closest_running_dir(direction_to_ball)


def rush_toward_ball(obs, controlled_player_pos, ball_pos):
    if distance(ball_pos, controlled_player_pos) < SPRINTING_SPEED:
        return run_toward_ball(obs, controlled_player_pos, ball_pos)

    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, steps=5)
    direction_to_future_ball = direction(future_ball_pos, controlled_player_pos)
    return get_closest_running_dir(direction_to_future_ball)


# ATTACK

def tiki_taka(obs, controlled_player_pos, controlled_player_dir, running_dir):
    if running_dir in [Action.TopRight, Action.BottomRight, Action.BottomLeft, Action.TopLeft]:
        return Action.ShortPass

    controlled_player_pos_y = controlled_player_pos[1]

    # if controlled_player_dir[0] > 0:
    #     if running_dir == Action.Right and is_offside(obs, obs['left_team'][PlayerRole.CentralFront.value][0]):
    #         if controlled_player_pos[1] >= 0:
    #             return Action.BottomRight
    #         else:
    #             return Action.TopRight
    #
    #     return Action.ShortPass
    #
    # for player in obs['right_team']:
    #     distance_to_player = distance(player, controlled_player_pos)
    #
    #     if distance_to_player < SAFE_DISTANCE:
    #         return Action.ShortPass

    # if is safe then turn forward:
    if Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    desired_empty_distance = 0.15
    actions = [Action.TopRight, Action.BottomRight]
    desired_actions = []
    for action in actions:
        direction_norm = action_to_dir(action)
        direction = direction_mul(direction_norm, desired_empty_distance)
        if not is_opp_in_area(obs, controlled_player_pos + direction):
            desired_actions.append(action)

    if not desired_actions:
        for action in [Action.TopLeft, Action.BottomLeft]:
            direction_norm = action_to_dir(action)
            direction = direction_mul(direction_norm, desired_empty_distance)
            if not is_opp_in_area(obs, controlled_player_pos + direction):
                desired_actions.append(action)

    if desired_actions:
        actions = desired_actions
    else:
        return Action.ShortPass

    if controlled_player_pos_y > HALF:
        if Action.BottomLeft in actions:
            return Action.BottomLeft
        else:
            return Action.BottomRight
    else:
        if Action.TopLeft in actions:
            return Action.TopLeft
        else:
            return Action.TopRight


def wing_run(obs, controlled_player_pos, running_dir):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    with open('./log', mode='a') as f:
        f.write(f'ball on wing: {controlled_player_pos_x} / {str(running_dir.value)} / {is_opp_in_area(obs, controlled_player_pos)} / {is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0])}')
        f.write('\n')

    if controlled_player_pos_x > 1 - BRAKING_DISTANCE:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

        if is_opp_in_area(obs, controlled_player_pos):
            goal_dir = get_closest_running_dir([1, 0])
            if goal_dir in obs['sticky_actions']:
                return goal_dir
            return Action.LongPass

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

    if running_dir != Action.Right:  # not in obs['sticky_actions']:
        return Action.Right
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    # TODO: is player in front then do else
    return Action.Right


def switch_side(obs, controlled_player_pos, controlled_player_dir, last_side='left'):
    # controlled player has the ball!
    controlled_player_pos_x = controlled_player_pos[0]
    current_running_action = get_closest_running_dir(controlled_player_dir)

    if controlled_player_pos_x > - 0.25:
        return protect_ball(obs, controlled_player_pos)

    if Action.Dribble not in obs['sticky_actions']:
        return Action.Dribble

    desired_actions = [Action.Bottom, Action.BottomRight, Action.BottomLeft] if last_side == 'left' else \
        [Action.Top, Action.TopRight, Action.TopLeft]

    if current_running_action in desired_actions:
        # running correct direction
        if is_dangerous(obs, controlled_player_pos, controlled_player_dir):
            desired_actions.remove(current_running_action)
            new_dir = random.choice(desired_actions)
            return new_dir
        else:
            if is_free_player_in_front(obs, controlled_player_pos, controlled_player_dir, preferred_side='right'):
                return Action.ShortPass
            else:
                return protect_ball(obs, controlled_player_pos)
    else:
        # running wrong direction
        if not is_dangerous(obs, controlled_player_pos,
                            [0, RUNNING_SPEED] if last_side == 'left' else [0, -RUNNING_SPEED]):
            return Action.Bottom if last_side == 'left' else Action.Top
        else:
            if is_free_player_in_front(obs, controlled_player_pos, controlled_player_dir, preferred_side='right'):
                return Action.ShortPass
            else:
                return protect_ball(obs, controlled_player_pos)


def protect_ball(obs, controlled_player_pos):
    controlled_player_pos_y = controlled_player_pos[1]

    if controlled_player_pos_y < -1 + SECTOR_SIZE:
        return Action.ShortPass

    if Action.Dribble not in obs['sticky_actions']:
        return Action.Dribble
    elif abs(controlled_player_pos_y) > WING:
        return Action.Left

    elif controlled_player_pos_y > HALF:
        return Action.Bottom
    else:
        return Action.Top


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


def dribble_into_empty_space(obs, controlled_player_pos):
    controlled_player_pos_y = controlled_player_pos[1]
    preferred_action = Action.TopRight if controlled_player_pos_y <= 0 else Action.BottomRight
    worse_action = Action.TopRight if controlled_player_pos_y > HALF else Action.BottomRight

    if not is_dangerous(obs, controlled_player_pos, direction_mul(action_to_dir(Action.Right), RUNNING_SPEED)):
        return Action.Right
    elif not is_dangerous(obs, controlled_player_pos, direction_mul(action_to_dir(preferred_action), RUNNING_SPEED)):
        return preferred_action
    elif not is_dangerous(obs, controlled_player_pos, direction_mul(action_to_dir(worse_action), RUNNING_SPEED)):
        return worse_action
    else:
        with open('./log', mode='a') as f:
            f.write(f'protecting ball, {preferred_action}, {direction_mul(action_to_dir(Action.Right), RUNNING_SPEED)}')
            f.write('\n')
        return protect_ball(obs, controlled_player_pos)


def play_9(obs, controlled_player_pos, running_dir, marking_defs):
    controlled_player_pos_x = controlled_player_pos[0]
    controlled_player_pos_y = controlled_player_pos[1]

    if running_dir in [Action.Top, Action.TopRight, Action.BottomRight, Action.Bottom]:
        return Action.LongPass

    # play long ball to the winger
    marking_defs_y = 0
    for marking_def in marking_defs:
        marking_defs_y += marking_def[1]

    if not marking_defs:  # pass to higher winger
        left_winger = obs['left_team'][PlayerRole.LeftMidfield.value]
        right_winger = obs['left_team'][PlayerRole.RIghtMidfield.value]
        if left_winger[0] > right_winger[0] and not is_offside(obs, left_winger[0]):
            if running_dir == Action.TopRight:  # in obs['sticky_actions']:
                return Action.LongPass
            else:
                return Action.TopRight
        elif not is_offside(obs, right_winger[0]):
            if running_dir == Action.BottomRight:  # in obs['sticky_actions']:
                return Action.LongPass
            else:
                return Action.BottomRight
        else:
            return Action.Right
    elif marking_defs_y / len(marking_defs) >= controlled_player_pos_y:
        if running_dir == Action.Top:
            if Action.TopRight in obs['sticky_actions']:
                return Action.LongPass
            else:
                return Action.TopRight
        elif running_dir == Action.TopRight:
            return Action.LongPass
        else:
            return Action.Top
    else:
        if running_dir == Action.Bottom:
            if Action.BottomRight in obs['sticky_actions']:
                return Action.LongPass
            else:
                return Action.BottomRight
        elif running_dir == Action.BottomRight:
            return Action.LongPass
        else:
            return Action.Bottom


def play_goalkeeper(obs, controlled_player_pos, controlled_player_dir):
    controlled_player_pos_y = controlled_player_pos[1]

    if abs(controlled_player_pos_y) > WING:
        if Action.Right not in obs['sticky_actions']:
            return Action.Right
        return Action.HighPass

    desired_empty_distance = 0.2
    action_sets = [[Action.Top, Action.Bottom], [Action.TopRight, Action.BottomRight]]
    desired_actions = []
    for actions in action_sets:
        for action in actions:
            direction_norm = action_to_dir(action)
            direction = [x * desired_empty_distance for x in direction_norm]
            if not is_opp_in_sector(obs, controlled_player_pos + direction):
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

    return Action.HighPass


# DEFENSE

def retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos):
    ball_height = ball_pos[2]

    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, 2)
    direction_to_ball = direction(ball_pos, controlled_player_pos)
    direction_to_future_ball = direction(future_ball_pos, controlled_player_pos)

    ball_speed = dir_distance(ball_dir)
    # if ball_speed > 0.5 * SPRINTING_SPEED:
    #     direction_to_ball = direction(future_ball_pos, controlled_player_pos)
    # else:
    #     direction_to_ball = direction(ball_pos, controlled_player_pos)

    if ball_height < 0.03:
        if ball_speed > SPRINTING_SPEED:
            direction_to_ball = direction_to_future_ball
        if direction_diff_deg(controlled_player_dir, direction_to_ball) < 90:
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

    return get_closest_running_dir(direction_to_future_ball)


def rush_to_defense(obs, controlled_player_pos, ball_pos):
    if Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint

    ball_dir = obs['ball_direction']
    future_ball_pos = get_future_pos(ball_pos, ball_dir, steps=3)

    mid_between_ball_and_goal = [(future_ball_pos[0] - 1) / 2, future_ball_pos[1]/2]

    direction_to_defense = [mid_between_ball_and_goal[0] - controlled_player_pos[0],
                            mid_between_ball_and_goal[1] - controlled_player_pos[1]]
    return get_closest_running_dir(direction_to_defense)
