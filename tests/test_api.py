import pytest

from jungle.demo_env import DemoMultiAgentEnv
from jungle.various_envs import EasyExit


# env_config: {
#                 "world_shape": [5, 5],
#                 "n_agents": 2,
#                 "max_episode_len": 10,
#                 "action_space": action_space,
#                 "goal_shift": goal_shift,
#             }

def test_demo_rl_loop():
    demo_env  = DemoMultiAgentEnv({"world_shape": [5, 5],
                                    "n_agents": 2,
                                    "max_episode_len": 10,
                                    "action_space": "discrete",
                                    "goal_shift": 1})

    print(demo_env.action_space)
    print("hi")

    hexa_env  = EasyExit({
                "grid_size": 11,
                "n_agents": 2})

    assert demo_env.cfg["n_agents"]  == 2
