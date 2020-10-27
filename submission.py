import operator
import traceback

import math
import numpy as np
from kaggle_environments.envs.football.helpers import *

from keydb import KeyDB

WING = 0.3
DB = KeyDB(host='localhost')


@human_readable_agent
def agent(obs):
    def rush_toward_ball(obs, controlled_player_pos, ball_pos):
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

        ball_dir = obs['ball_direction']
        future_ball_pos = get_future_ball_pos(ball_pos, ball_dir, steps=3)
        direction_to_future_ball = list(map(operator.sub, future_ball_pos, controlled_player_pos))
        return get_closest_running_dir(direction_to_future_ball)

    def attack(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        # Does the player we control have the ball?
        if obs['ball_owned_player'] == obs['active']:
            controlled_player_pos_y = controlled_player_pos[1]
            if controlled_player_pos_y > WING:
                DB.set('football__last_side', 'right')
                return switch_side(obs, controlled_player_pos, controlled_player_dir, last_side='right')
            elif controlled_player_pos_y < -WING:
                DB.set('football__last_side', 'left')
                return switch_side(obs, controlled_player_pos, controlled_player_dir, last_side='left')
            else:
                last_side = DB.get('football__last_side')
                return switch_side(obs, controlled_player_pos, controlled_player_dir, last_side=last_side)
        else:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

    def retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        ball_height = ball_pos[2]

        ball_dir = obs['ball_direction']
        future_ball_pos = get_future_ball_pos(ball_pos, ball_dir)

        # direction_to_ball = list(map(operator.sub, ball_pos, controlled_player_pos))
        direction_to_future_ball = list(map(operator.sub, future_ball_pos, controlled_player_pos))

        if ball_height < 0.03 and direction_diff_deg(controlled_player_dir, direction_to_future_ball) < 90:
            distance_to_ball_x = direction_to_future_ball[0]
            distance_to_ball_y = direction_to_future_ball[1]
            distance_to_ball = math.sqrt(
                distance_to_ball_x * distance_to_ball_x + distance_to_ball_y * distance_to_ball_y)

            if distance_to_ball <= 0.025:
                return Action.Slide

        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
        return get_closest_running_dir(direction_to_future_ball)

    def switch_side(obs, controlled_player_pos, controlled_player_dir, last_side='left'):
        # controlled player has the ball!
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint

        current_running_action = get_closest_running_dir(controlled_player_dir)
        desired_actions = [Action.Bottom, Action.BottomRight, Action.BottomLeft] if last_side == 'left' else \
            [Action.TOP, Action.TopRight, Action.TopLeft]

        if current_running_action in desired_actions:
            # running correct direction
            if is_free_player_in_front(controlled_player_pos, controlled_player_dir, preferred_side='right'):
                return Action.ShortPass
            else:
                return cruise_or_lash(obs, controlled_player_dir, current_running_action)
        else:
            # running wrong direction
            if not is_dangerous(obs, controlled_player_pos, [0, 1] if last_side == 'left' else [0, -1]):
                return Action.Bottom if last_side == 'left' else Action.Top
            else:
                if is_free_player_in_front(controlled_player_pos, controlled_player_dir, preferred_side='right'):
                    return Action.ShortPass
                else:
                    return cruise_or_lash(obs, controlled_player_dir, current_running_action)

    def cruise_or_lash(obs, controlled_player_dir, current_running_action):
        if is_dangerous(obs, controlled_player_pos, controlled_player_dir):
            # TODO: or dribble
            return Action.LongPass
        else:
            return current_running_action

    def is_free_player_in_front(controlled_player_pos, controlled_player_dir, preferred_side):
        return True

    def is_dangerous(obs, controlled_player_pos, controlled_player_dir):
        controlled_player_pos_x = controlled_player_pos[0]
        controlled_player_dir_x = controlled_player_dir[0]
        controlled_player_pos_y = controlled_player_pos[1]
        controlled_player_dir_y = controlled_player_dir[1]

        for player in obs['right_team']:
            player_x = player[0]
            player_y = player[1]
            if player_x - controlled_player_pos_x - controlled_player_dir_x * 5 < 0 and \
                    player_y - controlled_player_pos_y - controlled_player_dir_y * 5 < 0:
                return True
        return False

    def give_ball_away():
        if controlled_player_dir[0] > 0:
            return Action.LongPass
        else:
            return Action.Right

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

    def get_future_ball_pos(ball_pos, ball_dir, steps=1):
        future_ball_pos = ball_pos + list(map(lambda x: x * steps, ball_dir))  # TODO: verify if ball dir is speed (change during step)
        return future_ball_pos

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

    ball_owned_team = obs['ball_owned_team']
    controlled_player_pos = obs['left_team'][obs['active']]
    controlled_player_dir = obs['left_team_direction'][obs['active']]

    try:
        ball_pos = obs['ball']
        if ball_owned_team == -1:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)
        elif ball_owned_team == 0:  # we have the ball
            return attack(obs, controlled_player_pos, controlled_player_dir, ball_pos)
        else:  # ball_owned_team = 1, opponents
            return defend(obs, controlled_player_pos, controlled_player_dir, ball_pos)
    except Exception as e:
        with open('./log', mode='a') as f:
            f.write(traceback.format_exc())
            f.write('\n')
        return Action.Idle

    # Make sure player is running.
    # if Action.Sprint not in obs['sticky_actions']:
    #     return Action.Sprint

    # Does the player we control have the ball?
    # if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
    #     # Shot if we are 'close' to the goal (based on 'x' coordinate).
    #     if controlled_player_pos[0] > 0.5:
    #         return Action.Shot
    #     # Run towards the goal otherwise.
    #     return Action.Right
