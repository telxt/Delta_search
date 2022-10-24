from subprocess import check_output
import torch
from collections import OrderedDict
import copy
import os
from task_list import QA_task_list as TASK_LIST

delta = 'adapter'
source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata'
save_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/average_all_tasks'
os.makedirs(save_ckpt_path, exist_ok=True)

if delta == 'adapter':
    state_list = []
    for task in TASK_LIST:
        path = source_ckpt_path + '/' + task + '/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    for i in range(len(state_list)):
        checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list)
        new_state_list.pop(i)

        print(i, len(new_state_list))

        #average
        unit_state_dict = OrderedDict()
        unit_state_dict['args'] = state_dict['args']
        unit_state_dict['adapter'] = {}
        unit_state_dict['current_state'] = state_dict['current_state']
        for (k, v) in state_dict['adapter'].items():
            weight = v - v
            for one_state in new_state_list:
                weight += one_state['adapter'][k]
            weight /= len(new_state_list)
            unit_state_dict['adapter'][k] = weight

        torch.save(unit_state_dict, checkpoint_path)

if delta == 'lora':
    state_list = []
    for task in TASK_LIST:
        path = source_ckpt_path + '/' + task + '/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    for i in range(len(state_list)):
        checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list)
        new_state_list.pop(i)

        print(i, len(new_state_list))

        #average
        unit_state_dict = OrderedDict()
        unit_state_dict['args'] = state_dict['args']
        unit_state_dict['lora'] = {}
        unit_state_dict['current_state'] = state_dict['current_state']
        for (k, v) in state_dict['lora'].items():
            weight = v - v
            for one_state in new_state_list:
                weight += one_state['lora'][k]
            weight /= len(new_state_list)
            unit_state_dict['lora'][k] = weight

        torch.save(unit_state_dict, checkpoint_path)

if delta == 'prefix':
    state_list = []
    for task in TASK_LIST:
        path = source_ckpt_path + '/' + task + '/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    for i in range(len(state_list)):
        checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list)
        new_state_list.pop(i)

        print(i, len(new_state_list))

        #average
        unit_state_dict = OrderedDict()
        unit_state_dict['args'] = state_dict['args']
        unit_state_dict['prefix'] = {}
        unit_state_dict['current_state'] = state_dict['current_state']
        for (k, v) in state_dict['prefix'].items():
            weight = v - v
            for one_state in new_state_list:
                weight += one_state['prefix'][k]
            weight /= len(new_state_list)
            unit_state_dict['prefix'][k] = weight

        torch.save(unit_state_dict, checkpoint_path)