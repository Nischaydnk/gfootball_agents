import traceback

from kaggle_environments.envs.football.helpers import *

from custom_agents.actions import *


@human_readable_agent
def agent(obs):

    def transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir):
        ball_pos_x = ball_pos[0]
        ball_dir_x = ball_dir[0]
        ball_dir_y = ball_dir[1]
        ball_height = ball_dir[2]
        ball_2d_dir = [ball_dir_x, ball_dir_y]
        goal_dir = get_closest_running_dir([1, 0])

        distance_from_goal = distance(ball_pos, [1, 0])
        if distance_from_goal <= SHOOTING_DISTANCE:
            goal_dir = get_closest_running_dir(controlled_player_dir)
            if running_dir != goal_dir:
                return goal_dir
            return Action.Shot

        # directional first touch
        ball_to_receiver_dir = direction(controlled_player_pos, ball_pos)
        if ball_height < 0.03 and direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) < 30:
            ball_speed = dir_distance(ball_2d_dir)
            distance_to_ball = distance(ball_pos, controlled_player_pos)
            if distance_to_ball < ball_speed and not is_opp_in_area(obs, controlled_player_pos):
                return dribble_into_empty_space(obs, controlled_player_pos)
            elif distance_to_ball < ball_speed:
                return protect_ball(obs, controlled_player_pos)
            else:
                return get_closest_running_dir(opposite_dir(ball_to_receiver_dir))

        # steering goalkeeper
        if int(obs['active']) == int(PlayerRole.GoalKeeper.value):
            goalie_pos = obs['left_team'][PlayerRole.GoalKeeper.value]
            d_goalie_to_ball = distance(goalie_pos, ball_pos)
            if d_goalie_to_ball < SAFE_DISTANCE:
                is_goalie_in_trouble = is_opp_in_sector(obs, ball_pos)
                if not is_goalie_in_trouble:
                    # TODO: what to do to NOT kick the ball?
                    return play_goalkeeper(obs, controlled_player_pos, controlled_player_dir)
            else:
                if is_opp_in_sector(obs, ball_pos):
                    return rush_to_defense(obs, controlled_player_pos, ball_pos)
                else:
                    return rush_toward_ball(obs, controlled_player_pos, ball_pos)

        if ball_dir_x > 0 and controlled_player_pos[0] < ball_pos_x:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)

        return lash(obs, controlled_player_pos, goal_dir)

    def attack(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        # Does the player we control have the ball?
        if obs['ball_owned_player'] == obs['active']:
            distance_from_goal = distance(controlled_player_pos, [1, 0])

            goal_dir = get_closest_running_dir(controlled_player_dir)
            if distance_from_goal <= SHOOTING_DISTANCE:
                if running_dir != goal_dir:
                    return goal_dir
                return Action.Shot

            # keeper out of goal
            goal_distance = distance(controlled_player_pos, [1, 0])
            keeper_distance = get_goalkeeper_distance(obs, controlled_player_pos)
            if keeper_distance < goal_distance / 2:
                if running_dir != goal_dir:
                    return goal_dir
                return Action.Shot

            if int(obs['ball_owned_player']) == int(PlayerRole.GoalKeeper.value):
                return play_goalkeeper(obs, controlled_player_pos, controlled_player_dir)

            goal_dir = get_closest_running_dir([1, 0])
            return lash(obs, controlled_player_pos, goal_dir)
        else:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        controlled_player_pos_x = controlled_player_pos[0]
        ball_pos_x = ball_pos[0]

        if ball_pos_x > HALF and is_attacker(obs):  # attacking roles
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

        elif ball_pos_x > HALF and controlled_player_pos_x > -LAST_THIRD:
            # TODO: or retrieve depending on density around ball
            return rush_to_defense(obs, controlled_player_pos, ball_pos)

        # if controlling wrong player then run opposite to ball to change to proper player
        # elif is_attacker(obs) and is_defender_behind(obs, controlled_player_pos):
        #     return Action.Right
        elif are_defenders_behind(obs, controlled_player_pos):
            return Action.Right

        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

    try:
        ball_pos = obs['ball']
        ball_dir = obs['ball_direction']
        game_mode = obs['game_mode']

        if game_mode == GameMode.Penalty:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            return Action.Shot

        elif game_mode == GameMode.GoalKick:
            goalie_pos = obs['left_team'][0]
            d_goalie_to_ball = distance(ball_pos, goalie_pos)
            if d_goalie_to_ball < 1:  # is OUR goal kick
                goalie_dir = obs['left_team_direction'][0]
                goalie_action = get_closest_running_dir(goalie_dir)
                if goalie_action in [Action.Top, Action.Bottom]:
                    return Action.ShortPass
                else:
                    return random.choice([Action.Top, Action.Bottom])

        elif game_mode == GameMode.Corner:
            return Action.HighPass

        elif game_mode == GameMode.FreeKick and ball_pos[0] < HALF:
            return random.choice([Action.Top, Action.Bottom, Action.ShortPass])

        elif game_mode != GameMode.Normal:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            return Action.ShortPass

        ball_owned_team = obs['ball_owned_team']
        controlled_player_pos = obs['left_team'][obs['active']]
        controlled_player_dir = obs['left_team_direction'][obs['active']]
        running_dir = get_closest_running_dir(controlled_player_dir)

        if ball_owned_team == -1:
            return transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)
        elif ball_owned_team == 0:  # we have the ball
            return attack(obs, controlled_player_pos, controlled_player_dir, ball_pos)
        else:  # ball_owned_team = 1, opponents
            return defend(obs, controlled_player_pos, controlled_player_dir, ball_pos)
    except Exception as e:
        with open('./error_log', mode='a') as f:
            f.write(traceback.format_exc())
            f.write('\n')
        return Action.Idle
