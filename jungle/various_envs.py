from jungle_env import JungleBase
from utils import ElementsEnv
from copy import deepcopy

class EasyExit(JungleBase):

    def __init__(self):
        self.add_agents()

    def _set_exits(self):
        self.exit_1 = self.select_random_exit()

        self.add_objects()

    def _set_elements(self):
        pass

    def reset(self):
        self.grid_env[:] = deepcopy(self._initial_grid)

        self._place_agents()
        self._assign_colors()
        self.add_objects()

        self.agents[0].reset()
        self.agents[1].reset()

        obs = {self.agents[0]: self.generate_agent_obs(self.agents[0]),
               self.agents[1]: self.generate_agent_obs(self.agents[1])}

        return obs

    def add_objects(self):
        self.add_object(ElementsEnv.EXIT_EASY, self.exit_1.coordinates)