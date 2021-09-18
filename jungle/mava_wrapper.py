import copy
from typing import Any, Dict, Iterator, List, Optional, Union
import sys, os 
sys.path.append(".")
sys.path.append("..")

import dm_env
#import gym
import numpy as np

# from acme import specs
# from acme.wrappers.gym_wrapper import _convert_to_spec


from mava import types
from mava.utils.wrapper_utils import (
    apply_env_wrapper_preprocessers,
    convert_dm_compatible_observations,
    convert_np_type,
    parameterized_restart,
)

from jungle_env import JungleBase
from mava.wrappers.env_wrappers import ParallelEnvWrapper



class MavaJungleWrapper(JungleBase):

    def __init__(self,
        environment: JungleBase):

        self._environment = environment
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
    """Resets the env.
        Returns:
            dm_env.TimeStep: dm timestep.
        """

        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

    #TODO look into how this discount spec works, whether we need it at this stage 

        discount_spec = self.discount_spec()
        observe = self._environment.reset()

        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
            }

    # TODO remember this also takes in observe from jungle 
    # _convert_observations takes in a dict of type string: np array ;
    # so I ahve to make sure my dict has the agent string and obs are a numpy array 

        observations = self._convert_observations(
            observe, {agent: False for agent in self.possible_agents}
            )

        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        return parameterized_restart(rewards, self._discounts, observations), env_extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.
        Args:
            actions (Dict[str, np.ndarray]): actions per agent.
        Returns:
            dm_env.TimeStep: dm timestep
        """

        if self._reset_next_step:
            return self.reset()

        #TODO: main interaction with my env happens here ; lets check all the output formats of:
        # observations
        # rewards
        # dones
        # infos 

        observations, rewards, dones, infos = self._environment.step(actions)

        rewards = self._convert_reward(rewards)
        observations = self._convert_observations(observations, dones)

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

    def env_done(self) -> bool:
      
        return not self.agents

    def _convert_reward(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Convert rewards to be dm_env compatible.
        Args:
            rewards (Dict[str, float]): rewards per agent.
        """
        rewards_spec = self.reward_spec()
        # Handle empty rewards
        if not rewards:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in self.possible_agents
            }
        else:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, reward)
                for agent, reward in rewards.items()
            }

        #TODO check the format of rewards. NOTES: in petting zoo eg https://www.pettingzoo.ml/environment_creation#example-custom-parallel-environment
        # they have a string initiation for possible agents. then in reset they initiate self.agents 
        # I have initiated self.agents in various_envs.py at the top of reset -- will need to run this through 
        # and see if it works 
        return rewards

    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> types.Observation:
        """Convert PettingZoo observation so it's dm_env compatible.
        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.
        Returns:
            types.Observation: dm compatible observations.
        """
        return convert_dm_compatible_observations(
            observes,
            dones,
            self._environment.action_spaces,
            self._environment.observation_spaces,
            self.env_done(),
            self.possible_agents,
        )




    


