import sys, os 
sys.path.append(".")
sys.path.append("..")
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env

from ray.tune.logger import pretty_print, DEFAULT_LOGGERS, TBXLogger

# from ray.tune.integration.wandb import WandbLogger

from jungle.demo_env import DemoMultiAgentEnv
from jungle.various_envs import EasyExit
from networks.model import Model
from ray.rllib.models import ModelCatalog
from networks.multi_trainer import MultiPPOTrainer
from networks.multi_action_dist import TorchHomogeneousMultiActionDistribution


def train(share_observations=True):
    ray.init()

    register_env("demo_env", lambda config: EasyExit(config))
    ModelCatalog.register_custom_model("model", Model)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )

    tune.run(
        MultiPPOTrainer,
        checkpoint_freq=1,
        keep_checkpoints_num=1,
        local_dir="/tmp",
        # loggers=DEFAULT_LOGGERS + (WandbLogger,),
        stop={"training_iteration": 2},
        config={
            "framework": "torch",
            "env": "demo_env",
            "kl_coeff": 0.0,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "train_batch_size": 10000,
            "rollout_fragment_length": 1250,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 16,
            "num_gpus": 0,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "lr": 5e-4,
            "gamma": 0.99,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "model": {
                "custom_model": "model",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "encoder_out_features": 8,
                    "shared_nn_out_features_per_agent": 8,
                    "value_state_encoder_cnn_out_features": 16,
                    "share_observations": share_observations,
                },
            },
            "logger_config": {
                "wandb": {
                    "project": "ray_multi_agent_trajectory",
                    "group": "a",
                    "api_key_file": "./wandb_api_key_file",
                }
            },
            "env_config": {
                "grid_size": 11,
                "n_agents": 2,
                #"max_episode_len": 10,
                #"action_space": action_space,
                #"goal_shift": goal_shift,
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RLLib multi-agent with shared NN demo."
    )
   
    #parser.add_argument(
        #"--disable_sharing",
        #action="store_true",
        #help="Do not instantiate shared central NN for sharing information",
    #)


    args = parser.parse_args()
    train(
        share_observations=True
        
    )
