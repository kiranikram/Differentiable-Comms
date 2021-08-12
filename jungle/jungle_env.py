import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle

from utils import  (ElementsEnv,Actions,Rewards, str_dict,
                          display_dict, MIN_SIZE_ENVIR, MAX_WOOD_LOGS, BLACK,
                          WHITE)


Exit = namedtuple('Exit', ['coordinates', 'surrounding_1', 'surrounding_2'])

class JungleBase(ABC,gym.Env):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.seed(1)

        self.size = env_config.get('grid_size')

        self.grid_env = np.ones((self.size, self.size), dtype=int)
        self.grid_env *= ElementsEnv.EMPTY.value

        # Placeholders for agents
        self.agents: List[Agent] = []

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

        self.agents = [Agent(range_observation = 4)]

