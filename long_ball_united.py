import traceback

from kaggle_environments.envs.football.helpers import *

from custom_agents.actions import *


@human_readable_agent
def agent(obs):

    def transition(obs, controlled_player_pos, ball_pos, ball_dir):
        ball_pos_x = ball_pos[0]
        ball_dir_x = ball_dir[0]
        ball_game_dir = get_closest_running_dir(ball_dir)
        if ball_pos_x >= 1 - SHOOTING_DISTANCE:
            return Action.Shot
        elif ball_game_dir == Action.Right:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)

        return Action.LongPass

    def attack(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        # Does the player we control have the ball?
        if obs['ball_owned_player'] == obs['active']:
            distance_from_goal_x = controlled_player_pos[0] - 1
            distance_from_goal_y = controlled_player_pos[1]
            distance_from_goal = math.sqrt(distance_from_goal_x * distance_from_goal_x + distance_from_goal_y * distance_from_goal_y)
            if distance_from_goal <= SHOOTING_DISTANCE:
                return Action.Shot

            return lash(obs, controlled_player_pos)
        else:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        controlled_player_pos_x = controlled_player_pos[0]
        ball_pos_x = ball_pos[0]

        if ball_pos_x > 0 and controlled_player_pos_x > -0.3:
            # TODO: or retrieve depending on density around ball
            return rush_to_defense(obs, controlled_player_pos, ball_pos)

        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

    game_mode = obs['game_mode']
    if game_mode == GameMode.Penalty:
        return Action.Shot
    elif game_mode != GameMode.Normal:
        return Action.ShortPass

    ball_owned_team = obs['ball_owned_team']
    controlled_player_pos = obs['left_team'][obs['active']]
    controlled_player_dir = obs['left_team_direction'][obs['active']]

    try:
        ball_pos = obs['ball']
        ball_dir = obs['ball_direction']
        if ball_owned_team == -1:
            return transition(obs, controlled_player_pos, ball_pos, ball_dir)
        elif ball_owned_team == 0:  # we have the ball
            return attack(obs, controlled_player_pos, controlled_player_dir, ball_pos)
        else:  # ball_owned_team = 1, opponents
            return defend(obs, controlled_player_pos, controlled_player_dir, ball_pos)
    except Exception as e:
        with open('./error_log', mode='a') as f:
            f.write(traceback.format_exc())
            f.write('\n')
        return Action.Idle
