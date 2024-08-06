import openpyxl
import pandas as pd
from collections import OrderedDict
import os
# class debug:
#     def __init__(self,debug: bool = False):
#
#         self.debug = debug
#         self.debug_data = {name: [] for name in self.debug_name} if debug else None
#
#
#     def record_debug_data(act, rew, obs):
#         debug_data = {name: [] for name in debug_name} if debug else None
#         # debug_data['state'].append(obs.tolist())
#         debug_data['action'].append(act.tolist())
#         debug_data['reward'].append(rew)
#
#     @property
#     def debug_name():
#         d = OrderedDict()
#         axises = ['x', 'y', 'z']
#         d['reward'] = ['rew' + '_' + str(i) for i in range(1)]
#         d['joint_act'] = ['act' + '_' + str(i) for i in range(11)]
#         # d['obs_state'] = ['obs' + '_' + str(i) for i in range(self.env.num_observations)]
#
#         return d

def initialize_excel(file_name):
    wb = openpyxl.Workbook()
    ws_action = wb.active
    ws_action.title = 'Action'
    ws_reward = wb.create_sheet(title='Reward')

    # 设置列标题
    # ws_action.append(['action_' + str(i + 1) for i in range(act.shape)])
    # ws_reward.append(['reward'])

    return wb,  ws_action, ws_reward
def append_to_excel(ws_action, ws_reward,action, reward):

    ws_action.append(action.tolist())
    ws_reward.append([reward])


def save_debug_data(wb,file_name):
    # 保存 Excel 文件
    wb.save(file_name)
    print(f"Data has been saved to {file_name}")


