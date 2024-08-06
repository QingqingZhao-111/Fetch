import time
import cv2
import importlib
import numpy as np
import os
from os.path import join, exists
import torch
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import parse_config
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import download_assets
from utils.utils import load_config,get_args
from utils.debug import initialize_excel,append_to_excel,save_debug_data

from utils.CombinedExtractor import CustomCombinedExtractor
import yaml




def play(args,selection="user", headless=False, short_exec=False):
    # video=False
    exp_dir = join('experiments',args.name)
    # exp_dir = join('experiments', 'example_experiment2')
    model_dir = join(exp_dir, 'model')
    debug_dir = join(exp_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    file_path=os.getcwd()
    config_file =file_path+("/config/fetch_rearrangement.yaml")
    config_data = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # data = {
    #     'state': [],
    #     'action': [],
    #     'reward': []
    # }




    if args.video:
        # args.render = True
        pic_folder = os.path.join(debug_dir, 'picture')
        folder = os.path.exists(pic_folder)
        if not folder:
            os.makedirs(pic_folder)
    if args.debug:
        file_name=os.path.join(debug_dir, 'debug.xlsx')
        wb, ws_action, ws_reward = initialize_excel(file_name)
    # Load the configuration
    # Improving visuals in the example (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True
    model_save_path=join(model_dir, 'policy')
    model = PPO.load(model_save_path)
    # config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
    env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
    max_iterations = 2 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        state=env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                with torch.no_grad():
                    act, _ = model.predict(state)
                # action = env.action_space.sample()
                state, reward, done, info = env.step(act)


                if args.debug:
                    append_to_excel(ws_action, ws_reward, action=act, reward=reward)

                if done:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
    save_debug_data(wb,file_name)
    env.close()

    # # Load the PPO model
    # model_path = join(model_dir, 'ppo_model.zip')
    # model = PPO.load(model_path, env=gym_env)
    #
    # print(f'--------------------------------------------------------------------------------------')
    # print(f'Start to evaluate policy `{exp_dir}`.')
    # cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    # for epoch in range(args.epochs):
    #     print(f'#The `{epoch + 1}st/(total {args.epochs} times)` rollout......................................')
    #     obs = gym_env.reset()
    #     for i in range(int(args.time / (cfg.sim.dt * cfg.pd_gains.decimation))):
    #         with torch.no_grad():
    #             act, _ = model.predict(obs, deterministic=True)
    #         obs, rew, done, info = gym_env.step(act)
    #         cur_reward_sum += torch.tensor(rew, dtype=torch.float, device=device)
    #         if args.video:
    #             if i >= 0:
    #                 filename = os.path.join(pic_folder, f"{i}.png")
    #                 env.render()  # Adjust this according to your rendering setup
    #                 cv2.imwrite(filename, env.render(mode='rgb_array'))
    #         if any(done):
    #             break
    #     print(f'Evaluation finished after {i} timesteps.')
    #     print('cur_reward_sum', cur_reward_sum)
    #     if args.debug:
    #         save_debug_data(debug_dir)
    #     if args.video:
    #         fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
    #         cap_fps = int(1 / (cfg.sim.dt * cfg.pd_gains.decimation))
    #         video_path = join(debug_dir, f'{args.name}.mp4')
    #         video = cv2.VideoWriter(video_path, fourcc, cap_fps, (1600, 900))
    #         file_lst = os.listdir(pic_folder)
    #         file_lst.sort(key=lambda x: int(x[:-4]))
    #         for filename in file_lst:
    #             img = cv2.imread(join(pic_folder, filename))
    #             video.write(img)
    #         video.release()
    #         shutil.rmtree(pic_folder)
    #         print(f'#The video has been saved into `{video_path}`.')
    # print(f'End of evaluation.')
    # print(f'--------------------------------------------------------------------------------------')
    #

if __name__ == '__main__':
    args = get_args()
    play(args)
