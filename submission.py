import traceback

from kaggle_environments.envs.football.helpers import *

from custom_agents.actions import *


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
                return stay_front_of_goal(obs, controlled_player_pos, ball_pos), 'transition goalkeeper 1'
            else:
                return receive_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir), 'transition goalkeeper 2'

        # is closest to ball
        if ball_speed < BALL_SPEED and distance_to_ball < SAFE_DISTANCE and not is_opp_in_area(obs, ball_2d_pos):
            return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir), 'transition closest'

        # stand in defense
        if ball_dir_x < -0.005 and not are_defenders_behind(obs, controlled_player_pos) and controlled_player_pos_x > LAST_THIRD:
            return rush_to_defense(obs, controlled_player_pos, ball_pos), 'transition rush_to_defense 1'

        elif ball_dir_x < -0.005 and not are_defenders_behind(obs, controlled_player_pos):
            return rush_to_stop_ball(obs, controlled_player_pos, ball_pos), 'transition rush_to_stop_ball 1'

        # chase ball on wing
        if abs(controlled_player_pos_y) > WING and direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) > 30:  # and controlled_player_pos[0] > HALF:
            if controlled_player_pos_x < 1 - BRAKING_DISTANCE_WING or distance_to_ball > 0.025:
                if (0 < controlled_player_pos_y < ball_pos_y or ball_pos_y < controlled_player_pos_y > 0) and abs(ball_to_receiver_dir[0]/ball_to_receiver_dir[1]) < 1.5:
                    return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir), 'transition wing 1'
                else:
                    return sprint_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir), 'transition wing 1'
            else:
                return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir), 'transition wing 2'

        distance_to_goal = distance([1, 0], ball_2d_pos)
        if distance_to_goal < SHORT_SHOOTING_DISTANCE and abs(controlled_player_pos_y) < SECTOR_SIZE:
            if ball_speed > BALL_SPEED and direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) <= 5 and ball_height < 0.15:
                return Action.Shot, 'transition shot'
            elif ball_speed > BALL_SPEED and ball_height < 0.15:
                ball_dir_action = get_closest_running_dir(opposite_dir(ball_to_receiver_dir))
                return random.choice([Action.Shot, ball_dir_action, ball_dir_action, ball_dir_action]), 'transition random shot'
            else:
                return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir), 'transition predict_ball 0'

        # directional first touch
        if direction_diff_deg(ball_2d_dir, ball_to_receiver_dir) <= 30:
            return react_to_incoming_ball(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir,
                                          ball_to_receiver_dir, running_dir)

        return predict_ball(obs, controlled_player_pos, controlled_player_dir, ball_2d_pos, ball_dir), 'transition predict_ball 2'

    def attack(obs, controlled_player_pos, controlled_player_dir, running_dir, ball_pos, ball_dir, modeled_action):
        ball_2d_pos = [ball_pos[0], ball_pos[1]]
        # Does the player we control have the ball?
        if int(obs['ball_owned_player']) == int(obs['active']):
            controlled_player_pos_x = controlled_player_pos[0]
            controlled_player_pos_y = controlled_player_pos[1]

            goal_dir = [1 - controlled_player_pos_x, -controlled_player_pos_y]
            goal_dir_action = get_closest_running_dir(goal_dir)
            distance_from_goal = dir_distance(goal_dir)

            if distance_from_goal <= SHORT_SHOOTING_DISTANCE and (controlled_player_pos_y == 0 or abs((1 - controlled_player_pos_x)/controlled_player_pos_y) >= SHORT_SHOOTING_ANGLE):
                return Action.Shot, 'attack shot short'

            if (distance_from_goal <= SHOOTING_DISTANCE and
                (controlled_player_pos_y == 0 or abs((1 - controlled_player_pos_x)/controlled_player_pos_y) >= SHOOTING_ANGLE)) or \
                    (Action.Sprint in obs['sticky_actions'] and distance_from_goal <= LONG_SHOOTING_DISTANCE and
                     (controlled_player_pos_y == 0 or abs((1 - controlled_player_pos_x)/controlled_player_pos_y) >= LONG_SHOOTING_ANGLE)):
                if goal_dir_action not in obs['sticky_actions']:
                    return goal_dir_action, 'attack shot long'

                if controlled_player_pos_x > 1 - SAFE_DISTANCE:
                    if Action.Sprint in obs['sticky_actions']:
                        return Action.ReleaseSprint, 'attack shot long'
                return Action.Shot, 'attack shot long'

            # keeper out of goal
            goal_distance = distance(controlled_player_pos, [1, 0])
            keeper_distance = get_goalkeeper_distance(obs, controlled_player_pos)
            if keeper_distance < goal_distance / 2:
                if running_dir != goal_dir_action and Action.Sprint not in obs['sticky_actions']:
                    return goal_dir_action
                # if running_dir != goal_dir_action and controlled_player_pos_x <= 1 - SAFE_DISTANCE:
                #     if Action.Sprint in obs['sticky_actions']:
                #         return Action.ReleaseSprint
                #     return goal_dir_action
                return Action.Shot, 'attack shot vs gk'

            if controlled_player_pos_x > 1 - SAFE_DISTANCE:
                if Action.Sprint in obs['sticky_actions']:
                    return Action.ReleaseSprint, 'attack last inches'
                if Action.Dribble not in obs['sticky_actions']:
                    return Action.Dribble, 'attack last inches'

                if controlled_player_pos_y > 0 and \
                        (running_dir != Action.Top or Action.Top not in obs['sticky_actions']):
                    return Action.Top, 'attack last inches'
                elif controlled_player_pos_y < 0 and \
                        (running_dir != Action.Bottom or Action.Bottom not in obs['sticky_actions']):
                    return Action.Bottom, 'attack last inches'

                # mod_obs = custom_convert_observation([obs], None)
                # action, _states = assist_cross_model.predict(mod_obs)
                # if action is not None:
                #     return Action(int(action) + 1), 'modeled cross'

                # if modeled_action is not None:
                #     try:
                #         return Action(int(modeled_action)), 'trained action'
                #     except:
                #         pass

                if controlled_player_pos_y > 0:
                    if is_dangerous(obs, controlled_player_pos, [0, -SAFE_DISTANCE]):
                        return cross(obs, running_dir), 'attack cross'
                    return Action.Top, 'attack last inches'
                elif controlled_player_pos_y < 0:
                    if is_dangerous(obs, controlled_player_pos, [0, SAFE_DISTANCE]):
                        return cross(obs, running_dir), 'attack cross'
                    return Action.Bottom, 'attack last inches'
                elif controlled_player_pos_y == 0:
                    return Action.Shot, 'attack last inches'
                else:  # should never reach here
                    return Action.Idle, 'attack last inches'

            is_one_on_one, marking_defs = is_1_on_1(obs, controlled_player_pos)

            if is_one_on_one and abs(controlled_player_pos_y) < WING:
                return finalize_action(obs, controlled_player_pos, goal_dir_action, modeled_action), 'attack finalize action'

            if abs(controlled_player_pos_y) > WING:
                return wing_play(obs, controlled_player_pos, running_dir, ball_pos, ball_dir, modeled_action), 'attack wing play'

            if controlled_player_pos_x > -LAST_THIRD and (is_9(obs) or is_mid_up_front(obs, controlled_player_pos)) and offside_line(obs) <= LAST_THIRD:
                return play_9(obs, controlled_player_pos, running_dir, marking_defs, goal_dir_action, modeled_action), 'attack play 9'

            if controlled_player_pos_x > LAST_THIRD:
                return finalize_action(obs, controlled_player_pos, goal_dir_action, modeled_action), 'attack finalize action'

            if controlled_player_pos_x > -LAST_THIRD:
                return midfield_play(obs, controlled_player_pos, running_dir, modeled_action), 'attack midfield play'

            # kick anywhere but own goal
            if controlled_player_pos_x <= SAFE_DISTANCE and controlled_player_pos_y <= POST:
                if running_dir == Action.Left or Action.Left in obs['sticky_actions']:
                    return Action.Right, 'attack from own goal'
                else:
                    return Action.HighPass, 'attack from own goal'

            if controlled_player_pos_x <= -LAST_THIRD:
                if obs['ball_owned_player'] in [PlayerRole.LeftBack, PlayerRole.RightBack]:
                    return protect_ball(obs, controlled_player_pos, running_dir), 'attack protect ball'
                else:
                    return center_back_play(obs, controlled_player_pos, running_dir, modeled_action), 'attack centerback play'

            # should never reach this
            return dribble_into_empty_space(obs, controlled_player_pos, running_dir), 'attack dribble'
        else:
            return walk_toward_ball(obs, controlled_player_pos, ball_2d_pos, ball_dir), 'attack run to ball'

    def defend(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir):
        controlled_player_pos_x = controlled_player_pos[0]
        # controlled_player_pos_y = controlled_player_pos[1]
        ball_pos_x = ball_pos[0]

        # ball is rightin front
        # if abs(controlled_player_pos[1] - ball_pos[1]) < SAFE_DISTANCE and \
        #         controlled_player_pos_x < ball_pos_x < controlled_player_pos_x + SECTOR_SIZE:
        #     return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos)

        if is_goalkeeper(obs):
            return stay_front_of_goal(obs, controlled_player_pos, ball_pos), 'defense goalkeeper'

        if ball_pos_x > LAST_THIRD and not is_attacker(obs):  # attacking roles
            return rush_to_defense(obs, controlled_player_pos, ball_pos), 'defense rush_to_defense'

        elif ball_pos_x > HALF and is_attacker(obs):  # attacking roles
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir), 'defense retrieve_ball_asap 1'

        elif ball_pos_x > HALF:  # and is_defender(obs):
            if are_defenders_behind(obs, controlled_player_pos):
                return rush_to_stop_ball(obs, controlled_player_pos, ball_pos, offside_trap=True), 'defense rush_to_stop_ball 0'
            else:
                return rush_to_stop_ball(obs, controlled_player_pos, ball_pos), 'defense rush_to_stop_ball 1'

        elif -1 + SECTOR_SIZE < ball_pos_x <= controlled_player_pos_x:  # and is_defender(obs):
            return rush_to_stop_ball(obs, controlled_player_pos, ball_pos), 'defense rush_to_stop_ball 2'

        elif ball_pos_x > -1 + SECTOR_SIZE and ball_pos_x > controlled_player_pos_x:
            if is_closest_to_ball(obs, controlled_player_pos, ball_2d_pos):
                retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)
            if are_defenders_behind(obs, controlled_player_pos):
                return control_attacker(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir), 'defense control_attacker'
            else:
                return rush_to_stop_ball(obs, controlled_player_pos, ball_pos), 'defense rush_to_stop_ball 3'

        # if controlling wrong player then run opposite to ball to change to proper player
        # elif are_defenders_behind(obs, controlled_player_pos):
        #     return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir)

        elif ball_pos_x <= -1 + SECTOR_SIZE:
            return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir), 'defense retrieve_ball_asap 2'

        return retrieve_ball_asap(obs, controlled_player_pos, controlled_player_dir, ball_pos, ball_dir), 'defense retrieve_ball_asap 3'

    try:
        ball_pos = obs['ball']
        ball_dir = obs['ball_direction']
        game_mode = obs['game_mode']
        ball_2d_pos = [ball_pos[0], ball_pos[1]]
        ball_to_goal_distance = distance(ball_2d_pos, [1, 0])
        controlled_player_pos = obs['left_team'][obs['active']]

        if game_mode != GameMode.Normal:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint, 'out of play'
            if Action.Sprint in obs['sticky_actions']:
                return Action.Dribble, 'out of play'

        if game_mode == GameMode.Penalty:
            if Action.Sprint in obs['sticky_actions']:
                return Action.ReleaseSprint
            return Action.Shot, 'penalty'

        elif game_mode == GameMode.GoalKick:
            return random.choice([Action.Top, Action.Bottom]), 'goal kick'

        elif game_mode == GameMode.Corner:
            if controlled_player_pos[1] > 0 and Action.Top not in obs['sticky_actions']:
                return Action.Top, 'corner'
            elif controlled_player_pos[1] < 0 and Action.Bottom not in obs['sticky_actions']:
                return Action.Bottom, 'corner'
            return Action.HighPass, 'corner'

        elif game_mode == GameMode.ThrowIn:
            if ball_pos[0] < LAST_THIRD:
                return random.choice([Action.Right, Action.ShortPass]), 'throw in'
            else:
                return Action.LongPass, 'throw in'

        elif game_mode == GameMode.FreeKick and ball_to_goal_distance <= FK_SHOOTING_DISTANCE:
            return Action.Shot, 'free kick'
        elif game_mode == GameMode.FreeKick and ball_pos[0] >= LAST_THIRD:
            return Action.ShortPass, 'free kick'
        elif game_mode == GameMode.FreeKick and ball_pos[0] < LAST_THIRD:
            return random.choice([Action.Top, Action.Bottom, Action.ShortPass]), 'free kick'

        elif game_mode == GameMode.KickOff:
            if Action.Sprint not in obs['sticky_actions']:
                return Action.Sprint, 'kick off'
            return Action.Left, 'kick off'

        elif game_mode != GameMode.Normal:
            return Action.ShortPass, 'unknown game mode?'

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
        with open('./error_log', mode='a') as f:
            f.write(traceback.format_exc())
            f.write('\n')
        return Action.Idle, 'error'
