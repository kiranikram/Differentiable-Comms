import copy
from typing import Any, Dict, Iterator, List, Optional, Union
import sys, os 
sys.path.append(".")
sys.path.append("..")

import dm_env
#import gym
import numpy as np

from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec


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
    # should be okay, just check it 

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

    def observation_spec(self) -> types.Observation:
        """Observation spec.
        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self.possible_agents:
            observation_specs[agent] = types.OLT(
                observation=_convert_to_spec(
                    self._environment.observation_spaces[agent]
                ),
                legal_actions=_convert_to_spec(self._environment.action_spaces[agent]),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.
        Returns:
            Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]: spec for actions.
        """
        action_specs = {}
        action_spaces = self._environment.action_spaces
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(action_spaces[agent])
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.
        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.
        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Extra data spec.
        Returns:
            Dict[str, specs.BoundedArray]: spec for extra data.
        """
        return {}

    #TODO we dont have this abiliti (to see if agents are alive -- need to see how to implement it )
    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).
        Returns:
            List: alive agents in env.
        """
        return self._environment.agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.
        Returns:
            List: all possible agents in env.
        """
        return self._environment.possible_agents

    @property
    def environment(self) -> JungleBase:
        """Returns the wrapped environment.
        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

    @property
    def current_agent(self) -> Any:
        """Current active agent.
        Returns:
            Any: current agent.
        """
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.
        Args:
            name (str): attribute.
        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name

    

    

    

    




    


