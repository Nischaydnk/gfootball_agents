import traceback

from kaggle_environments.envs.football.helpers import *

from custom_agents.actions import *


@human_readable_agent
def agent(obs):

    def transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir):
        ball_pos_x = ball_pos[0]
        ball_dir_x = ball_dir[0]
        ball_dir_y = ball_dir[1]
        ball_2d_pos = [ball_pos[0], ball_pos[1]]
        ball_2d_dir = [ball_dir_x, ball_dir_y]

        controlled_player_pos_x = controlled_player_pos[0]

        if abs(controlled_player_pos[1]) > WING:
            if controlled_player_pos[0] > HALF:
                return rush_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir)

        # steering goalkeeper
        if is_goalkeeper(obs):
            return rush_keeper(obs, controlled_player_pos, ball_2d_pos, ball_dir)
            # goalie_pos = obs['left_team'][PlayerRole.GoalKeeper.value]
            # d_goalie_to_ball = distance(goalie_pos, ball_pos)
            # if d_goalie_to_ball < SAFE_DISTANCE:
            #     is_goalie_in_trouble = is_opp_in_sector(obs, ball_pos)
            #     if not is_goalie_in_trouble:
            #         # TODO: what to do to NOT kick the ball?
            #         return play_goalkeeper(obs, controlled_player_pos, controlled_player_dir)
            # else:
            #     if is_opp_in_sector(obs, ball_pos):
            #         return rush_to_defense(obs, controlled_player_pos, ball_pos)
            #     else:
            #         return rush_toward_ball(obs, controlled_player_pos, ball_pos)

        # directional first touch
        ball_to_receiver_dir = direction(controlled_player_pos, ball_2d_pos)
        if direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) <= 30:
            return react_to_incoming_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir,
                                          ball_to_receiver_dir, running_dir)

        if ball_dir_x > 0 and controlled_player_pos_x < ball_pos_x:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

        elif ball_pos_x > LAST_THIRD:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

        elif ball_pos_x > HALF:
            if is_opp_in_area(obs, controlled_player_pos):
                if controlled_player_dir[0] > 0 and is_dangerous(obs, controlled_player_pos, [0.05, 0]):
                    return Action.LongPass
                elif controlled_player_dir[0] > 0 or running_dir == Action.Left:
                    return rush_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)
                else:
                    return Action.ShortPass
            else:
                return rush_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)
        else:  # ball flying in our half
            return rush_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

    def attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_pos, ball_dir):
        # Does the player we control have the ball?
        if int(obs['ball_owned_player']) == int(obs['active']):
            controlled_player_pos_x = controlled_player_pos[0]
            controlled_player_pos_y = controlled_player_pos[1]

            goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
            goal_dir_action = get_closest_running_dir(goal_dir)
            distance_from_goal = dir_distance(goal_dir)

            if distance_from_goal <= SHOOTING_DISTANCE and abs(controlled_player_pos_y) < SECTOR_SIZE:
                if distance_from_goal <= SHORT_SHOOTING_DISTANCE:
                    return Action.Shot

                # if running_dir != goal_dir_action:
                #     return goal_dir_action
                if running_dir != goal_dir_action and controlled_player_pos_x <= 1 - SAFE_DISTANCE:
                    if Action.Sprint in obs['sticky_actions']:
                        return Action.ReleaseSprint
                    return goal_dir_action
                return Action.Shot

            # keeper out of goal
            goal_distance = distance(controlled_player_pos, [1, 0])
            keeper_distance = get_goalkeeper_distance(obs, controlled_player_pos)
            if keeper_distance < goal_distance / 2:
                # if running_dir != goal_dir_action:
                #     return goal_dir_action
                # if running_dir != goal_dir_action and controlled_player_pos_x <= 1 - SAFE_DISTANCE:
                #     if Action.Sprint in obs['sticky_actions']:
                #         return Action.ReleaseSprint
                #     return goal_dir_action
                return Action.Shot

            if controlled_player_pos_x > 1 - SAFE_DISTANCE:
                if Action.Sprint in obs['sticky_actions']:
                    return Action.ReleaseSprint
                if Action.Dribble not in obs['sticky_actions']:
                    return Action.Dribble

                if controlled_player_pos_y > 0 and\
                        (running_dir != Action.Top or Action.Top not in obs['sticky_actions']):
                    return Action.Top
                elif controlled_player_pos_y < 0 and \
                        (running_dir != Action.Bottom or Action.Bottom not in obs['sticky_actions']):
                    return Action.Bottom

                if controlled_player_pos_y > HALF:
                    if is_dangerous(obs, controlled_player_pos, [0, -SAFE_DISTANCE]):
                        return cross(obs)
                    return Action.Top
                elif controlled_player_pos_y < -HALF:
                    if is_dangerous(obs, controlled_player_pos, [0, SAFE_DISTANCE]):
                        return cross(obs)
                    return Action.Bottom
                else:
                    return Action.Shot

            is_one_on_one, marking_defs = is_1_on_1(obs, controlled_player_pos)

            if is_one_on_one and abs(controlled_player_pos_y) < WING:
                return finalize_action(obs, controlled_player_pos, goal_dir_action)

            if is_goalkeeper(obs):
                return goalkeeper_play(obs, controlled_player_pos, controlled_player_dir, running_dir)

            if abs(controlled_player_pos_y) >= WING:
                return wing_play(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)

            is_9 = int(obs['ball_owned_player']) == int(PlayerRole.CentralFront.value)
            if controlled_player_pos_x > -LAST_THIRD and is_9:  # or (0 < controlled_player_pos_x < 0.2 and -0.15 < controlled_player_pos_y < 0.15):
                return play_9(obs, controlled_player_pos, running_dir, marking_defs, goal_dir_action)

            if controlled_player_pos_x > LAST_THIRD:
                return finalize_action(obs, controlled_player_pos, goal_dir_action)

            if controlled_player_pos_x > -LAST_THIRD:
                return midfield_play(obs, controlled_player_pos, running_dir)

            if controlled_player_pos_x <= -LAST_THIRD:
                if obs['ball_owned_player'] in [PlayerRole.LeftBack, PlayerRole.RightBack]:
                    return protect_ball(obs, controlled_player_pos, running_dir)
                else:
                    return center_back_play(obs, controlled_player_pos, running_dir)

            # should never reach this
            return dribble_into_empty_space(obs, controlled_player_pos, running_dir)
        else:
            return run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir):
        controlled_player_pos_x = controlled_player_pos[0]
        # controlled_player_pos_y = controlled_player_pos[1]
        ball_pos_x = ball_pos[0]

        # ball is rightin front
        # if abs(controlled_player_pos[1] - ball_pos[1]) < SAFE_DISTANCE and \
        #         controlled_player_pos_x < ball_pos_x < controlled_player_pos_x + SECTOR_SIZE:
        #     return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

        if ball_pos_x > HALF and is_attacker(obs):  # attacking roles
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

        elif ball_pos_x > HALF:  # and is_defender(obs):
            return rush_to_defense(obs, controlled_player_pos, ball_pos)

        elif ball_pos_x > -SECTOR_SIZE and is_defender(obs):
            return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)

        elif ball_pos_x > -SECTOR_SIZE and is_goalkeeper(obs):
            return stay_front_of_goal(obs, controlled_player_pos, ball_pos)

        elif ball_pos_x > -SECTOR_SIZE:
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

        elif ball_pos_x <= -SECTOR_SIZE:
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
        # if controlling wrong player then run opposite to ball to change to proper player
        # elif are_defenders_behind(obs, controlled_player_pos):
        #     return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

    try:
        ball_pos = obs['ball']
        ball_dir = obs['ball_direction']
        game_mode = obs['game_mode']
        ball_2d_pos = [ball_pos[0], ball_pos[1]]
        ball_to_goal_distance = distance(ball_2d_pos, [1, 0])

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
            goalie_pos = obs['left_team'][0]
            d_goalie_to_ball = distance(ball_2d_pos, goalie_pos)
            if d_goalie_to_ball < 1:  # is OUR goal kick
                goalie_dir = obs['left_team_direction'][0]
                goalie_action = get_closest_running_dir(goalie_dir)
                if goalie_action in [Action.Top, Action.Bottom]:
                    return Action.ShortPass
                else:
                    return random.choice([Action.Top, Action.Bottom])

        elif game_mode == GameMode.Corner:
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

        elif game_mode != GameMode.Normal:
            return Action.ShortPass

        ball_owned_team = obs['ball_owned_team']
        controlled_player_pos = obs['left_team'][obs['active']]
        controlled_player_dir = obs['left_team_direction'][obs['active']]
        running_dir = get_closest_running_dir(controlled_player_dir)

        if ball_owned_team == -1:
            return transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)
        elif ball_owned_team == 0:  # we have the ball
            return attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_2d_pos, ball_dir)
        else:  # ball_owned_team = 1, opponents
            return defend(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
    except Exception as e:
        with open('./error_log', mode='a') as f:
            f.write(traceback.format_exc())
            f.write('\n')
        return Action.Idle
