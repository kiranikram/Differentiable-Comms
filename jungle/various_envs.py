import sys, os 
sys.path.append(".")
sys.path.append("..")
from jungle.jungle_env import JungleBase
from jungle.jungledemo_env import JungleDemoEnv
from jungle.utils import ElementsEnv
from copy import deepcopy
import gym

class EasyExit(JungleDemoEnv, gym.Env):

    

    def _set_exits(self):
        self.exit_1 = self.select_random_exit()

        self.add_objects()

    def _set_elements(self):
        pass

    def reset(self):
        print("well we get  here ")
        self.grid_env[:] = deepcopy(self._initial_grid)

        self._place_agents()
        self._assign_colors()
        self.add_objects()

        self.agents[0].reset()
        self.agents[1].reset()

        obs = {self.agents[0]: self.generate_agent_obs(self.agents[0]),
               self.agents[1]: self.generate_agent_obs(self.agents[1])}

        obs = self.postprocess_obs_data(obs)

        return obs

    def add_objects(self):
        self.add_object(ElementsEnv.EXIT_EASY, self.exit_1.coordinates)