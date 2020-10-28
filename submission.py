import traceback

from kaggle_environments.envs.football.helpers import *
from keydb import KeyDB

from custom_agents.actions import *

DB = KeyDB(host='localhost')


@human_readable_agent
def agent(obs):

    def transition(obs, controlled_player_pos, ball_pos, ball_dir):
        ball_pos_x = ball_pos[0]
        ball_dir_x = ball_dir[0]

        if Action.Dribble in obs['sticky_actions']:
            return Action.ReleaseDribble

        if ball_pos_x >= 1 - SHOOTING_DISTANCE:
            return Action.Shot
        elif ball_dir_x > 0:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)
        elif ball_pos_x > 0.3:
            if distance(ball_pos, controlled_player_pos) > 0.1:
                return rush_toward_ball(obs, controlled_player_pos, ball_pos)
            else:
                return Action.LongPass
        elif ball_pos_x > -0.3:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            if is_opp_in_area(obs, ball_pos):
                return Action.ShortPass
            else:
                return run_toward_ball(obs, controlled_player_pos, ball_pos)
        else:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            if is_opp_in_area(obs, ball_pos):
                return Action.HighPass
            else:
                return run_toward_ball(obs, controlled_player_pos, ball_pos)

    def attack(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        # Does the player we control have the ball?
        if obs['ball_owned_player'] == obs['active']:
            distance_from_goal_x = controlled_player_pos[0] - 1
            distance_from_goal_y = controlled_player_pos[1]
            distance_from_goal = math.sqrt(distance_from_goal_x * distance_from_goal_x + distance_from_goal_y * distance_from_goal_y)
            if distance_from_goal <= SHOOTING_DISTANCE:
                return Action.Shot

            controlled_player_pos_x = controlled_player_pos[0]
            controlled_player_pos_y = controlled_player_pos[1]

            if int(obs['ball_owned_player']) == int(PlayerRole.GoalKeeper.value):
                return play_goalkeeper(obs, controlled_player_pos, controlled_player_dir)

            if abs(controlled_player_pos_y) > WING and controlled_player_pos_x > 0.3:
                return wing_run(obs, controlled_player_pos)
            else:
                # return Action.ShortPass
                return tiki_taka(obs, controlled_player_pos, controlled_player_dir)

            # controlled_player_pos_y = controlled_player_pos[1]
            # if controlled_player_pos_y > WING:
            #     DB.set('football__last_side', 'right')
            #     return wing_run(obs, controlled_player_pos)
            #     # return switch_side(obs, controlled_player_pos, controlled_player_dir, last_side='right')
            # elif controlled_player_pos_y < -WING:
            #     DB.set('football__last_side', 'left')
            #     return wing_run(obs, controlled_player_pos)
            #     # return switch_side(obs, controlled_player_pos, controlled_player_dir, last_side='left')
            # else:
            #     last_side = DB.get('football__last_side')
            #     if not last_side:
            #         last_side = 'left'
            #     return switch_side(obs, controlled_player_pos, controlled_player_dir, last_side=last_side)
        else:
            return run_toward_ball(obs, controlled_player_pos, ball_pos)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        controlled_player_pos_x = controlled_player_pos[0]
        ball_pos_x = ball_pos[0]

        if ball_pos_x > 0 and controlled_player_pos_x > -0.3:
            # TODO: or retrieve depending on density around ball
            return rush_to_defense(obs, controlled_player_pos, ball_pos)

        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

    game_mode = obs['game_mode']
    if game_mode == GameMode.Penalty:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
        return Action.Shot
    elif game_mode != GameMode.Normal:
        if Action.Sprint in obs['sticky_actions']:
            return Action.ReleaseSprint
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
