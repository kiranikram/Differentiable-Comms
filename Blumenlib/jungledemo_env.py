import sys, os 
sys.path.append(".")
sys.path.append("..")
from typing import Tuple, Union, List

from abc import ABC, abstractmethod
from copy import deepcopy
import random
import math

from collections import namedtuple


import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from jungle.utils import  (ElementsEnv,Actions,Rewards, str_dict,
                          display_dict, MIN_SIZE_ENVIR, MAX_WOOD_LOGS, BLACK,
                          WHITE, action_conversion_dict)

from jungle.agent import Agent


Exit = namedtuple('Exit', ['coordinates', 'surrounding_1', 'surrounding_2'])
class JungleDemoEnv(gym.Env, EzPickle):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.seed(1)

        self.size = env_config.get('grid_size')
        self.grid_env = np.ones((self.size, self.size), dtype=int)
        self.grid_env *= ElementsEnv.EMPTY.value

        self.agents = [Agent(i,range_observation = 4)  for i in range(env_config.get('n_agents'))]

        # Set starting_positions
        pos_1 = int((self.size - 1) / 2), int((self.size - 1) / 2 - 1)
        angle_1 = 3
        self._starting_coordinates_1 = pos_1, angle_1

        pos_2 = int((self.size - 1) / 2), int((self.size - 1) / 2 + 1)
        angle_2 = 0
        self._starting_coordinates_2 = pos_2, angle_2

        # Set borders of environment
        self._set_boundaries()
        # Set elements
        self._set_elements()
        # Set Exits
        self._calculate_exit_coordinates()
        self._set_exits()
        # Save the initial grid to reset to original state
        self._initial_grid = deepcopy(self.grid_env)

        self._place_agents(random_position=True)
        self._assign_colors()

        self.observation_space = spaces.Dict(
            {
                "agents": spaces.Tuple(
                    (
                        spaces.Box(
                            low=-1,
                            high=1,
                            shape=(258,),
                        ),
                    )
                    * env_config.get("n_agents"))
                
                #"state": spaces.Box(
                    #low=0.0, high=1.0, shape=self.cfg["world_shape"] + [2]
                #),
            }
        )

        agent_action_space = spaces.Discrete(12)
        self.action_space = spaces.Tuple((agent_action_space,) * 2)

        #self.action_space = spaces.Tuple((agent_action_space,) * self.cfg["n_agents"])

        #self.agents = [Agent(i,range_observation = 4)  for i in range(env_config.get('n_agents'))]
        self.reset()

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

   

    def step(self, actions):
        actions_white = actions[0]
        actions_black = actions[1]
        
        #observations = [
            #agent.step(action) for agent, action in zip(self.agents, actions)
        #]
        #take  out agent class

        _white_actions = action_conversion_dict.get(actions_white)
        _black_actions = action_conversion_dict.get(actions_black)

        black_fwd = _black_actions[0]
        black_rot = _black_actions[1]
        black_climb = _black_actions[2]

        black_dict = {Actions.FORWARD: black_fwd, Actions.ROTATE: black_rot, Actions.CLIMB: black_climb}

        white_fwd = _white_actions[0]
        white_rot = _white_actions[1] 
        white_climb = _white_actions[2]

        white_dict = {Actions.FORWARD: white_fwd, Actions.ROTATE: white_rot, Actions.CLIMB: white_climb}

        actions_dict = {'white': white_dict,
                        'black': black_dict}

        obs, rewards, done, info = self.step_in_jungle(actions_dict)
        #format of obs to come out of here should be list with two numpy arrays inside 

        obs = {"agents": tuple(observations)}
        #blumenkamps :
        #global_state = np.zeros(self.cfg["world_shape"] + [2], dtype=np.uint8)
        #for agent in self.agents:
            #global_state[int(agent.pose[Y]), int(agent.pose[X]), 0] = 1
            #global_state[int(agent.goal[Y]), int(agent.goal[X]), 1] = 1

        return obs, all_rewards, done, info

    def step_in_jungle(self, actions):

        # First Physical move
        if not self.agents[0].done:
            rew_0 = self.move(self.agents[0], actions['white'])
            agent_0_climbs = actions['white'].get(Actions.CLIMB, 0)
        else:
            rew_0 = 0
            agent_0_climbs = False

        if not self.agents[1].done:
            rew_1 = self.move(self.agents[1], actions['black'])
            agent_1_climbs = actions['black'].get(Actions.CLIMB, 0)
        else:
            rew_1 = 0
            agent_1_climbs = False

        # If None are done, check interactions btween agents
        if self.agents[0].position == self.agents[1].position and self.agents[
                0].position:

            row, col = self.agents[0].position

            # TREE
            if self.grid_env[row, col] == ElementsEnv.TREE.value:

                # If they are on a tree they cut it
                self.grid_env[row, col] = ElementsEnv.EMPTY.value

                # one of them only gets the log
                if random.random() > 0.5:
                    self.agents[1].wood_logs += 1
                else:
                    self.agents[0].wood_logs += 1

                # But both have neg reward from the effort
                rew_0 += Rewards.REWARD_CUT_TREE.value
                rew_1 += Rewards.REWARD_CUT_TREE.value

            # RIVER
            if self.grid_env[row, col] == ElementsEnv.RIVER.value:

                # If they have enough logs they build a bridge
                if self.agents[0].wood_logs + self.agents[
                        1].wood_logs == 2 * MAX_WOOD_LOGS:
                    self.agents[0].wood_logs = 0
                    self.agents[1].wood_logs = 0
                    self.grid_env[row, col] = ElementsEnv.EMPTY.value

            # CLIMB Behavior if they are on the same cell
            if agent_0_climbs and not agent_1_climbs:
                self.agents[0].on_shoulders = True
                rew_1 += Rewards.REWARD_CARRYING.value

            elif agent_1_climbs and not agent_0_climbs:
                self.agents[1].on_shoulders = True
                rew_0 += Rewards.REWARD_CARRYING.value

            elif agent_1_climbs and agent_0_climbs:
                rew_0 += Rewards.REWARD_FELL.value
                rew_1 += Rewards.REWARD_FELL.value

        # If not on the same cell
        else:

            # If agent was on shoulders, but other agent moved:
            if self.agents[1].on_shoulders:
                self.agents[1].on_shoulders = False
                rew_1 += Rewards.REWARD_FELL.value

            if self.agents[0].on_shoulders:
                self.agents[0].on_shoulders = False
                rew_0 += Rewards.REWARD_FELL.value

        # Apply environment rules
        done_1 = self.agents[1].done
        if not done_1:
            rew, done_1 = self.apply_rules(self.agents[1])
            rew_1 += rew

        done_0 = self.agents[0].done
        if not done_0:
            rew, done_0 = self.apply_rules(self.agents[0])
            rew_0 += rew

        # All rewards and terminations are now calculated
        rewards = {self.agents[1]: rew_1, self.agents[0]: rew_0}

        # Now we calculate the observations
        obs = {}

        if not self.agents[0].done:
            obs[self.agents[0]] = self.generate_agent_obs(self.agents[0])
        else:
            obs[self.agents[0]] = None

        if not self.agents[1].done:
            obs[self.agents[1]] = self.generate_agent_obs(self.agents[1])
        else:
            obs[self.agents[1]] = None

        obs = self.postprocess_obs_data(obs)
        print('OBS in step')
        print(obs)

        info = {"rewards": rewards}
        all_rewards = sum(rewards.values())

        dones = {self.agents[0]: done_0, self.agents[1]: done_1}

        dones = self.postprocess_dones(done)

        self.agents[0].done = done_0
        self.agents[1].done = done_1

        return obs, all_rewards, dones , info

    def move(self, agent, actions_dict):

        reward = 0

        #action_dict = actions.get(agent, {})
        rotation = action_dict.get(Actions.ROTATE, 0)
        forward = action_dict.get(Actions.FORWARD, 0)

        agent.angle += rotation

        row, col = agent.position
        current_cell = self.cell_type(row, col)

        # If we don't move forward nothing happens
        if forward == 0:
            return 0

        # Else we see where we go
        row_new, col_new = self.get_proximal_coordinate(row, col, agent.angle)

        next_cell = self.cell_type(row_new, col_new)

        # If we were on a boulder, we can move to non-obstacles
        # We collide only if we go toward an obstacle.
        if current_cell == ElementsEnv.BOULDER.value:

            if next_cell == ElementsEnv.OBSTACLE.value:
                reward = Rewards.REWARD_COLLISION.value
                row_new, col_new = row, col

        # If we were on the ground, we move unless we face boulder or obstacle
        else:

            # Check if next cell is an obstacle, we don't move
            if next_cell == ElementsEnv.OBSTACLE.value:
                reward = Rewards.REWARD_COLLISION.value
                row_new, col_new = row, col

            # Check if next cell is a boulder
            elif next_cell == ElementsEnv.BOULDER.value:

                # If not on shoulders, we collide
                if not agent.on_shoulders:
                    reward = Rewards.REWARD_COLLISION.value
                    row_new, col_new = row, col

                # Else we move to boulders
                # and starting then we can go from boulder to boulder

        # Whatever happens, if we move forward, we are not on shoulders anymore
        agent.on_shoulders = False

        # Now that we now if we can move or not, we change position
        agent.position = row_new, col_new

        return reward

    def generate_agent_obs(self, agent):

        visual_obs = self._generate_full_observation(agent)

        if not (agent.on_shoulders
                or self.grid_env[agent.position] == ElementsEnv.BOULDER.value):
            visual_obs = self._filter_observations(visual_obs)

        # Add other agent relative angle if it is seen
        if agent is self.agents[0]:
            other_agent = self.agents[1]
        else:
            other_agent = self.agents[0]

        relative_angle = (other_agent.angle - agent.angle) % 6

        flat_visual_obs = []
        for line_obs in visual_obs:
            flat_visual_obs += line_obs

        if ElementsEnv.AGENT.value not in flat_visual_obs:
            relative_angle = -1

        obs_dict = {
            'visual': flat_visual_obs,
            'other_agent_angle': relative_angle,
            'color': agent.color
        }

        return obs_dict

    def _generate_full_observation(self, agent):

        grid_copy = deepcopy(self.grid_env)

        if agent == self.agents[0] and self.agents[1].position:
            row, col = self.agents[1].position
            grid_copy[row, col] = ElementsEnv.AGENT.value

        elif agent == self.agents[1] and self.agents[0].position:
            row, col = self.agents[0].position
            grid_copy[row, col] = ElementsEnv.AGENT.value

        obs = [[grid_copy[agent.position]]]

        # iterate over range
        for obs_range in range(1, agent.range_observation):

            line_obs = []

            row, col = agent.position
            angle = agent.angle

            # go to start
            for _ in range(obs_range):
                row, col, _ = self.get_next_cell(row, col, (angle - 1) % 6)

            if 0 <= row < self.size and 0 <= col < self.size:
                line_obs.append(grid_copy[row, col])
            else:
                line_obs.append(ElementsEnv.EMPTY.value)

            # move first segment
            for _ in range(obs_range):
                row, col, _ = self.get_next_cell(row, col, (angle + 1) % 6)

                if 0 <= row < self.size and 0 <= col < self.size:
                    line_obs.append(grid_copy[row, col])
                else:
                    line_obs.append(ElementsEnv.EMPTY.value)

            # move second segment
            for _ in range(obs_range):
                row, col, _ = self.get_next_cell(row, col, (angle + 2) % 6)

                if 0 <= row < self.size and 0 <= col < self.size:
                    line_obs.append(grid_copy[row, col])
                else:
                    line_obs.append(ElementsEnv.EMPTY.value)
            obs.append(line_obs)

        return obs

    def one_hot_obs(self, obs):
        env_elements = len(ElementsEnv) + 1
        one_hot_obs = np.eye(16)[obs]
        one_hot_obs = list(one_hot_obs.flat)
        return one_hot_obs

    def postprocess_obs_data(self,obs):

        if not obs[self.agents[0]] is None:
            white = obs[self.agents[0]]
            white_vis = self.one_hot_obs(white['visual'])
            white_oae = [white['other_agent_angle'] / self.size]
            white_color = [white['color'] / self.size]
            white_obs = white_vis + white_oae + white_color
            white_obs = np.array(white_obs)

        else:
            white_obs = np.zeros(258)  # should be length of obs

        if not obs[self.agents[1]] is None:
            black = obs[self.agents[1]]
            black_vis = self.one_hot_obs(black['visual'])
            black_oae = [black['other_agent_angle'] / self.size]
            black_color = [black['color'] / self.size]
            black_obs = black_vis + black_oae + black_color
            black_obs = np.array(black_obs)
            
        else:
            black_obs = np.zeros(258)

        # here  we  are  converting it to DEMO ENV  format  
        obs = [white_obs,black_obs]
        #obs = tuple(obs)

        return obs

    def postprocess_dones(self,done):
        white_done = done[self.agents[0]]
        black_done = done[self.agents[1]]

        if white_done and black_done:
            new_done = dict({"__all__": True})
        elif white_done and not black_done:
            new_done = dict({"__all__": False})
        elif black_done and not white_done:
            new_done = dict({"__all__": False})
        else:
            new_done = dict({"__all__": False})

        return new_done

    def _set_boundaries(self):

        # place outside walls
        self.grid_env[:, 0] = ElementsEnv.OBSTACLE.value
        self.grid_env[:, -1] = ElementsEnv.OBSTACLE.value
        self.grid_env[0, :] = ElementsEnv.OBSTACLE.value
        self.grid_env[-1, :] = ElementsEnv.OBSTACLE.value

        # add corners
        for row in range(2, self.size - 2, 2):
            self.grid_env[row, 1] = ElementsEnv.OBSTACLE.value
            
            
    @abstractmethod
    def _set_exits(self):
        pass

    @abstractmethod
    def _set_elements(self):
        pass

    def _calculate_exit_coordinates(self):

        self._exits = []

        self._exit_top_left = Exit((1, 1), (1, 2), (2, 2))
        self._exits.append(self._exit_top_left)

        self._exit_bottom_left = Exit((self.size - 2, 1), (self.size - 2, 2),
                                      (self.size - 3, 2))
        self._exits.append(self._exit_bottom_left)

        self._exit_top_right = Exit((1, self.size - 2), (1, self.size - 3),
                                    (2, self.size - 2))
        self._exits.append(self._exit_top_right)

        self._exit_bottom_right = Exit((self.size - 2, self.size - 2),
                                       (self.size - 2, self.size - 3),
                                       (self.size - 3, self.size - 2))
        self._exits.append(self._exit_bottom_right)

    def get_random_empty_location(self):

        grid = deepcopy(self.grid_env)

        # Avoid locations where agent is
        for agent in self.agents:
            if agent.position:
                grid[agent.position] = ElementsEnv.OBSTACLE.value

        # Avoid starting positions
        grid[self._starting_coordinates_1[0]] = ElementsEnv.OBSTACLE.value
        grid[self._starting_coordinates_2[0]] = ElementsEnv.OBSTACLE.value

        # Take all empty cells
        rows, cols = np.where(grid == ElementsEnv.EMPTY.value)

        # return one of them
        index = random.randint(0, len(rows) - 1)

        assert grid[rows[index], cols[index]] == ElementsEnv.EMPTY.value

        return rows[index], cols[index]

    def select_random_exit(self):
        """ Picks a random exit. """

        random.shuffle(self._exits)

        if not self._exits:
            raise ValueError('All exits have already been selected')

        return self._exits.pop(0)

    def _place_agents(self, random_position=True):

        if not random_position:
            self.agents[0].position, self.agents[
                0].angle = self._starting_coordinates_1
            self.agents[1].position, self.agents[
                1].angle = self._starting_coordinates_2
            return

        if random.random() > 0.5:
            self.agents[0].position, self.agents[
                0].angle = self._starting_coordinates_1
            self.agents[1].position, self.agents[
                1].angle = self._starting_coordinates_2

        else:
            self.agents[0].position, self.agents[
                0].angle = self._starting_coordinates_2
            self.agents[1].position, self.agents[
                1].angle = self._starting_coordinates_1

    def _assign_colors(self):

        if random.random() > 0.5:
            self.agents[0].color = BLACK
            self.agents[1].color = WHITE

        else:
            self.agents[0].color = WHITE
            self.agents[1].color = BLACK

    def reset(self):
        print("we reach reset")

        # Reset grid to initial state
        self.grid_env[:] = deepcopy(self._initial_grid)

        self._place_agents()
        self._assign_colors()

        self.agents[0].reset()
        self.agents[1].reset()

        obs = {
            self.agents[0]: self.generate_agent_obs(self.agents[0]),
            self.agents[1]: self.generate_agent_obs(self.agents[1])
        }

        obs = self.postprocess_obs_data(obs)

        return obs

    def apply_rules(self, agent):

        rew = 0

        # If on a tree, cut log
        agent_cuts = self.cutting_tree(agent)
        if agent_cuts:
            agent.wood_logs += 1
            rew += Rewards.REWARD_CUT_TREE.value

        # If on a river, drown
        agent_drowns = self.on_a_river(agent)
        if agent_drowns:
            rew += Rewards.REWARD_DROWN.value

        # If comes to boulder and not on shoulders, bumps into boulder
        bumps_boulder = self.hits_boulder(agent)
        if bumps_boulder:
            rew += Rewards.REWARD_COLLISION.value

        # If on an exit, receive reward and is done
        rew_exit, agent_exits = self.exits(agent)
        rew += rew_exit

        agent_done = agent_exits or agent_drowns

        return rew, agent_done

    def cutting_tree(self, agent):
        row, col = agent.position

        if self.cell_type(row, col) == ElementsEnv.TREE.value:
            # If they are on a tree they cut it
            self.grid_env[row, col] = ElementsEnv.EMPTY.value
            return True
        return False

    def on_a_river(self, agent):
        row, col = agent.position
        if self.cell_type(row, col) == ElementsEnv.RIVER.value:
            return True
        return False

    def hits_boulder(self, agent):
        row, col = agent.position
        if self.cell_type(row, col) == ElementsEnv.BOULDER.value:
            return True
        return False

    def exits(self, agent):
        row, col = agent.position
        current_cell = self.cell_type(row, col)

        done = True

        if current_cell == ElementsEnv.EXIT_BLACK.value and agent.color == BLACK:
            reward = Rewards.REWARD_EXIT_VERY_HIGH.value

        elif current_cell == ElementsEnv.EXIT_BLACK.value and agent.color == WHITE:
            reward = Rewards.REWARD_EXIT_LOW.value

        elif current_cell == ElementsEnv.EXIT_WHITE.value and agent.color == WHITE:
            reward = Rewards.REWARD_EXIT_VERY_HIGH.value

        elif current_cell == ElementsEnv.EXIT_WHITE.value and agent.color == BLACK:
            reward = Rewards.REWARD_EXIT_LOW.value

        elif current_cell == ElementsEnv.EXIT_EASY.value:
            reward = Rewards.REWARD_EXIT_AVERAGE.value

        elif current_cell == ElementsEnv.EXIT_DIFFICULT.value:
            reward = Rewards.REWARD_EXIT_HIGH.value

        # if we are not on an exit
        else:
            reward = 0
            done = False

        return reward, done

    @staticmethod
    def get_proximal_coordinate(row, col, angle):

        row_new, col_new = row, col

        if angle == 0:
            col_new += 1
        elif angle == 1:
            row_new -= 1
            col_new += row % 2
        elif angle == 2:
            row_new -= 1
            col_new += row % 2 - 1
        elif angle == 3:
            col_new -= 1
        elif angle == 4:
            row_new += 1
            col_new += row % 2 - 1
        else:
            row_new += 1
            col_new += row % 2

        return row_new, col_new

    @staticmethod
    def _filter_observations(obs):
        " Replace all occlusions with -1 "

        for index_line in range(1, len(obs) - 1):

            line_obs = obs[index_line]

            for i in range(len(line_obs)):

                # if there is an occlusion
                if line_obs[i] == -1 or (
                        line_obs[i] != ElementsEnv.EMPTY.value
                        and line_obs[i] != ElementsEnv.RIVER.value):

                    middle = int((len(line_obs) - 1) / 2)

                    # if it is on extremas
                    if i == middle:
                        obs[index_line + 1][i + 1] = -1

                    # if in angle, occludes 2 cells
                    elif i < middle:
                        obs[index_line + 1][i] = -1
                        obs[index_line + 1][i + 1] = -1

                    elif i > middle:
                        obs[index_line + 1][i + 1] = -1
                        obs[index_line + 1][i + 2] = -1

                    else:
                        raise ValueError

        return obs

    @staticmethod
    def _filter_observations(obs):
        " Replace all occlusions with -1 "

        for index_line in range(1, len(obs) - 1):

            line_obs = obs[index_line]

            for i in range(len(line_obs)):

                # if there is an occlusion
                if line_obs[i] == -1 or (
                        line_obs[i] != ElementsEnv.EMPTY.value
                        and line_obs[i] != ElementsEnv.RIVER.value):

                    middle = int((len(line_obs) - 1) / 2)

                    # if it is on extremas
                    if i == middle:
                        obs[index_line + 1][i + 1] = -1

                    # if in angle, occludes 2 cells
                    elif i < middle:
                        obs[index_line + 1][i] = -1
                        obs[index_line + 1][i + 1] = -1

                    elif i > middle:
                        obs[index_line + 1][i + 1] = -1
                        obs[index_line + 1][i + 2] = -1

                    else:
                        raise ValueError

        return obs

    def cell_type(self, x, y):
        return self.grid_env[x, y]

    def add_object(self, item, coords):
        r = coords[0]
        c = coords[1]
        self.grid_env[r, c] = item.value

    def get_next_cell(self, row: int, col: int,
                      angle: int) -> Tuple[int, int, int]:

        row_new, col_new = row, col

        if angle == 0:
            col_new += 1
        elif angle == 1:
            row_new -= 1
            col_new += row % 2
        elif angle == 2:
            row_new -= 1
            col_new += row % 2 - 1
        elif angle == 3:
            col_new -= 1
        elif angle == 4:
            row_new += 1
            col_new += row % 2 - 1
        else:
            row_new += 1
            col_new += row % 2

        if 0 <= row_new < self.size and 0 <= col_new < self.size:
            next_cell = self.grid_env[int(row_new), int(col_new)]
        else:
            next_cell = 0

        return row_new, col_new, next_cell

    

    

    

    

    

    

   
