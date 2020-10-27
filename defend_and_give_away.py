import operator

import math
import numpy as np
from kaggle_environments.envs.football.helpers import *


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
        return give_ball_away()

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

    def retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        ball_height = ball_pos[2]

        ball_dir = obs['ball_direction']
        future_ball_pos = get_future_ball_pos(ball_pos, ball_dir)

        # direction_to_ball = list(map(operator.sub, ball_pos, controlled_player_pos))
        direction_to_future_ball = list(map(operator.sub, future_ball_pos, controlled_player_pos))

        if ball_height < 0.03 and direction_diff_rad(controlled_player_dir, direction_to_future_ball) < 90:
            distance_to_ball_x = direction_to_future_ball[0]
            distance_to_ball_y = direction_to_future_ball[1]
            distance_to_ball = math.sqrt(
                distance_to_ball_x * distance_to_ball_x + distance_to_ball_y * distance_to_ball_y)

            if distance_to_ball <= 0.025:
                return Action.Slide

        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
        return get_closest_running_dir(direction_to_future_ball)

    def is_free_player_in_front(controlled_player_pos, controlled_player_dir, preferred_side):
        return True

    def is_dangerous(obs, controlled_player_pos, controlled_player_dir):
        return False

    def give_ball_away():
        if controlled_player_dir[0] > 0:
            return Action.LongPass
        else:
            return Action.Right

    def direction_diff_rad(dir1, dir2):
        dir1 = np.array(dir1)
        dir2 = np.array(dir2)
        unit_vector_1 = dir1 / np.linalg.norm(dir1)
        unit_vector_2 = dir2 / np.linalg.norm(dir2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        return np.rad2deg(np.arccos(dot_product))

    def get_future_ball_pos(ball_pos, ball_dir, steps=1):
        future_ball_pos = ball_pos + list(map(lambda x: x * 0.05 * steps, ball_dir))
        return future_ball_pos

    def get_closest_running_dir(running_dir):
        dir_x = running_dir[0]
        dir_y = running_dir[1]
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
            f.write(str(e))
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
