import logging
import os
from typing import Callable
from datetime import datetime
import igibson
from igibson.envs.igibson_env import iGibsonEnv
from utils.utils import load_config,get_args
from utils.CombinedExtractor import CustomCombinedExtractor

from torch.utils.tensorboard import SummaryWriter
from os.path import join
try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for PointNav task.
"""


class CustomCallback(BaseCallback):
    """
    Custom callback for logging additional information at each step.
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # This is called at each step
        step_info = {
            'num_timesteps': self.num_timesteps,
            'learning_rate': self.model.lr_schedule(self.num_timesteps),
            'n_updates': self.model._n_updates,
        }
        print(f"Step: {step_info['num_timesteps']}, Learning Rate: {step_info['learning_rate']}, Updates: {step_info['n_updates']}")
        return True
def train(args,selection="user", headless=False, short_exec=False):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """

    print('train start')
    print("*" * 80 + "\nDescription:" + train.__doc__ + "*" * 80)
    file_path=os.getcwd()
    config_file =file_path+("/config/fetch_rearrangement.yaml")
    config = load_config(file_path+("/config/ppo.yaml"))
    # task_dir = "task1"
    # tensorboard_log_dir = os.path.join(task_dir, "tensorboard_logs")

    # tensorboard_log_dir = "log_dir"
    num_environments = 2 if not short_exec else 1
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = join('experiments', args.name)
    # exp_dir = join('experiments', 'example_experiment2')
    model_dir = join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    all_model_dir = join(exp_dir, 'model', 'all')
    os.makedirs(all_model_dir, exist_ok=True)
    log_dir = join(exp_dir, 'log')
    # model_save_path = os.path.join(model_dir, "ckpt")
    # if os.path.exists(log_dir):
    #     for file in os.listdir(log_dir):
    #         os.remove(join(log_dir, file))
    # writer = SummaryWriter(log_dir, flush_secs=10)

    num_steps_per_env = config['ppo']['n_steps']
    num_learning_iterations = 10  # 设置学习迭代次数

    # Function callback to create environments
    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=os.path.join(igibson.configs_path, config_file),
                mode="headless",
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
            )
            # env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # Multiprocess
    env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
    env = VecMonitor(env)

    # Create a new environment for evaluation
    eval_env = iGibsonEnv(
        config_file=os.path.join(igibson.configs_path, config_file),
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )


    # Load PPO parameters from config
    ppo_params = config['ppo']
    # policy_kwargs = config.get('policy_kwargs', {})

    # # Obtain the arguments/parameters for the policy and create the PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    # os.makedirs(tensorboard_log_dir, exist_ok=True)
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
    # print(model.policy)
    model = PPO(
        "MultiInputPolicy",  # 根据环境选择合适的 Policy 类型
        env,
        learning_rate=ppo_params['learning_rate'],
        n_steps=ppo_params['n_steps'],
        batch_size=ppo_params['batch_size'],
        n_epochs=ppo_params['n_epochs'],
        gamma=ppo_params['gamma'],
        gae_lambda=ppo_params['gae_lambda'],
        clip_range=ppo_params['clip_range'],
        ent_coef=ppo_params['ent_coef'],
        vf_coef=ppo_params['vf_coef'],
        max_grad_norm=ppo_params['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir
    )

    total_timesteps = 0
    for it in range(num_learning_iterations):

        custom_callback = CustomCallback()
        model.learn(total_timesteps=num_steps_per_env * env.num_envs, reset_num_timesteps=False,callback=custom_callback)
        total_timesteps += num_steps_per_env * env.num_envs

        model.save(os.path.join(model_dir, 'policy'))
        if it % 10 == 0:  # Save every 10 iterations
            model.save(os.path.join(all_model_dir, f'policy_{it}'))

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        # writer.add_scalar('Train/mean_reward', mean_reward, it)

        print(f"Iteration {it}: Mean reward: {mean_reward:.2f}")
    # # Random Agent, evaluation before training
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1)
    # print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    #
    # # Train the model for the given number of steps
    # total_timesteps = 10 if short_exec else 10
    #
    # custom_callback = CustomCallback()
    # model.learn(total_timesteps,callback=custom_callback)
    #
    # # Evaluate the policy after training
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=2)
    # print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    #
    # # Save the trained model and delete it
    # # model.save("ckpt")
    # model.save(model_save_path)
    # # del model
    #
    # # Reload the trained model from file
    # # model = PPO.load("ckpt")
    # model = PPO.load(model_save_path)
    #
    #
    # # Evaluate the trained model loaded from file
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=2)
    # print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    args = get_args()
    train(args)