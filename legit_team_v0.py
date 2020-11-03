import traceback

from kaggle_environments.envs.football.helpers import *

from custom_agents.actions import *



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
        if running_dir == Action.Left:
            return Action.Right
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


@human_readable_agent
def agent(obs):

    def transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir):
        ball_pos_x = ball_pos[0]
        ball_dir_x = ball_dir[0]
        ball_dir_y = ball_dir[1]
        ball_height = ball_dir[2]
        ball_2d_dir = [ball_dir_x, ball_dir_y]

        controlled_player_pos_x = controlled_player_pos[0]
        controlled_player_pos_y = controlled_player_pos[1]

        goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
        goal_dir_action = get_closest_running_dir(goal_dir)

        distance_from_goal = distance(ball_pos, [1, 0])
        if distance_from_goal <= SHOOTING_DISTANCE and abs(controlled_player_pos_y) < SECTOR_SIZE:
            if running_dir != goal_dir_action:
                return goal_dir_action
            return Action.Shot

        if abs(controlled_player_pos[1]) > WING:
            if controlled_player_pos[0] > HALF:
                return wing_run(obs, controlled_player_pos, running_dir, ball_pos)

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

        if Action.Dribble in obs['sticky_actions']:
            return Action.ReleaseDribble

        # steering goalkeeper
        if int(obs['active']) == int(PlayerRole.GoalKeeper.value):
            goalie_pos = obs['left_team'][PlayerRole.GoalKeeper.value]
            d_goalie_to_ball = distance(goalie_pos, ball_pos)
            if d_goalie_to_ball < SAFE_DISTANCE:
                is_goalie_in_trouble = is_opp_in_sector(obs, ball_pos)
                if not is_goalie_in_trouble:
                    # TODO: what to do to NOT kick the ball?
                    return goalkeeper_play(obs, controlled_player_pos, controlled_player_dir)
            else:
                if is_opp_in_sector(obs, ball_pos):
                    return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)
                else:
                    return rush_toward_ball(obs, controlled_player_pos, ball_pos)

        if ball_dir_x > 0 and controlled_player_pos[0] < ball_pos_x:
            return rush_toward_ball(obs, controlled_player_pos, ball_pos)
        elif ball_pos_x > LAST_THIRD:
            if distance(ball_pos, controlled_player_pos) > SAFE_DISTANCE:
                return rush_toward_ball(obs, controlled_player_pos, ball_pos)
            elif is_opp_in_area(obs, controlled_player_pos):
                if controlled_player_dir[0] > 0 and is_dangerous(obs, controlled_player_pos, [0.05, 0]):
                    return Action.LongPass
                elif controlled_player_dir[0] > 0:
                    return dribble_into_empty_space(obs, controlled_player_pos)
                else:
                    return tiki_taka(obs, controlled_player_pos, controlled_player_dir, running_dir)
            else:
                return dribble_into_empty_space(obs, controlled_player_pos)
        elif ball_pos_x > -LAST_THIRD:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            if is_opp_in_area(obs, ball_pos):
                return protect_ball(obs, controlled_player_pos)  # return Action.ShortPass
            else:
                return run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)
        else:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            if is_opp_in_area(obs, ball_pos):
                return protect_ball(obs, controlled_player_pos)  # return Action.HighPass
            else:
                return run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

    def attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_pos):
        # Does the player we control have the ball?
        if int(obs['ball_owned_player']) == int(obs['active']):
            controlled_player_pos_x = controlled_player_pos[0]
            controlled_player_pos_y = controlled_player_pos[1]

            goal_dir = [1 - controlled_player_pos_x, - controlled_player_pos_y]
            goal_dir_action = get_closest_running_dir(goal_dir)
            distance_from_goal = dir_distance(goal_dir)

            if distance_from_goal <= SHOOTING_DISTANCE and abs(controlled_player_pos_y) < SECTOR_SIZE:
                if running_dir != goal_dir_action:
                    return goal_dir_action
                return Action.Shot

            # keeper out of goal
            goal_distance = distance(controlled_player_pos, [1, 0])
            keeper_distance = get_goalkeeper_distance(obs, controlled_player_pos)
            if keeper_distance < goal_distance / 2:
                if running_dir != goal_dir_action:
                    return goal_dir_action
                return Action.Shot

            is_one_on_one, marking_defs = is_1_on_1(obs, controlled_player_pos)

            if is_one_on_one and abs(controlled_player_pos_y) < SECTOR_SIZE:
                if running_dir != goal_dir_action:
                    return goal_dir_action
                else:
                    if Action.Sprint not in obs['sticky_actions']:
                        return Action.Sprint
                    # TODO: dribble the goalie
                    return Action.Right

            if int(obs['ball_owned_player']) == int(PlayerRole.GoalKeeper.value):
                return goalkeeper_play(obs, controlled_player_pos, controlled_player_dir)

            if int(obs['ball_owned_player']) == int(PlayerRole.CentralFront.value):  # or (0 < controlled_player_pos_x < 0.2 and -0.15 < controlled_player_pos_y < 0.15):
                return play_9(obs, controlled_player_pos, running_dir, marking_defs, goal_dir_action)

            if controlled_player_pos_x > 1 - SAFE_DISTANCE:
                if Action.Dribble not in obs['sticky_actions']:
                    return Action.Dribble

                if controlled_player_pos_y > HALF:
                    return Action.Top
                else:
                    return Action.Bottom

            if abs(controlled_player_pos_y) > WING:
                if controlled_player_pos_x > LAST_THIRD:
                    return wing_run(obs, controlled_player_pos, running_dir, ball_pos)
                elif running_dir == Action.Right and is_dangerous(obs, controlled_player_pos, [SAFE_DISTANCE, 0]):
                    return Action.LongPass
                elif running_dir == Action.Right:
                    return Action.Right
                elif is_opp_in_area(obs, controlled_player_pos):
                    return protect_ball(obs, controlled_player_pos)
                else:
                    return dribble_into_empty_space(obs, controlled_player_pos)

            if controlled_player_pos_x > LAST_THIRD:
                return dribble_into_empty_space(obs, controlled_player_pos)

            if controlled_player_pos_x < -LAST_THIRD:
                return protect_ball(obs, controlled_player_pos)

            # is_safe = not is_opp_in_area(obs, controlled_player_pos)
            # if is_safe and controlled_player_dir[0] > 0:  # already running forward
            #     return dribble_into_empty_space(obs, controlled_player_pos)
            # elif is_safe:
            #     return Action.Right

            return tiki_taka(obs, controlled_player_pos, controlled_player_dir, running_dir)

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
            return run_toward_ball(obs, controlled_player_pos, ball_pos, ball_dir)

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos):
        controlled_player_pos_x = controlled_player_pos[0]
        ball_pos_x = ball_pos[0]

        if ball_pos_x > HALF and is_attacker(obs):  # attacking roles
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

        elif ball_pos_x < LAST_THIRD and controlled_player_pos_x > HALF:
            # TODO: or retrieve depending on density around ball
            return rush_to_stop_ball(obs, controlled_player_pos, ball_pos)

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

        elif game_mode == GameMode.FreeKick and ball_pos[0] < LAST_THIRD:
            return random.choice([Action.Top, Action.Bottom, Action.ShortPass])
        elif game_mode == GameMode.FreeKick and ball_pos[0] >= LAST_THIRD:
            return Action.Shot

        elif game_mode != GameMode.Normal:
            return Action.ShortPass

        ball_owned_team = obs['ball_owned_team']
        controlled_player_pos = obs['left_team'][obs['active']]
        controlled_player_dir = obs['left_team_direction'][obs['active']]
        running_dir = get_closest_running_dir(controlled_player_dir)

        if ball_owned_team == -1:
            return transition(obs, controlled_player_pos, running_dir, ball_pos, ball_dir)
        elif ball_owned_team == 0:  # we have the ball
            return attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_pos)
        else:  # ball_owned_team = 1, opponents
            return defend(obs, controlled_player_pos, controlled_player_dir, ball_pos)
    except Exception as e:
        with open('./error_log', mode='a') as f:
            f.write(traceback.format_exc())
            f.write('\n')
        return Action.Idle
