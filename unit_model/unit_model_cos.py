from numpy import average
import torch
from numpy import average
from collections import OrderedDict
import copy
import os
import sys
sys.path.append('..')
from task_list import QA_task_list as TASK_LIST

temperature = 0.001
tuned_64shot_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/random'
source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata'
save_ckpt_path = '/home/lvxingtai/lxt/delta_search_code/checkpoints/nouse'

os.makedirs(save_ckpt_path, exist_ok=True)

state_list_64 = []
for task in TASK_LIST:
    path = tuned_64shot_ckpt_path + '/' + task + '/checkpoint-best.pt'
    state_dict = torch.load(path)
    state_list_64.append(state_dict)

state_list_full = []
for task in TASK_LIST:
    path = source_ckpt_path + '/' + task + '/checkpoint-best.pt'
    state_dict = torch.load(path)
    state_list_full.append(state_dict)

cos_sim = torch.nn.CosineSimilarity(dim=0)
for i in range(len(state_list_64)):
    checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'

    #get_weight
    weight_list = []
    state_dict1 = state_list_64[i]

    new_state_list = copy.deepcopy(state_list_full)
    new_state_list.pop(i)

    for j in range(len(new_state_list)):
        state_dict2 = new_state_list[j]
        sim_list = []
        for adapter_layer in state_dict1['adapter']:
            tensor1 = state_dict1['adapter'][adapter_layer].view(-1)
            tensor2 = state_dict2['adapter'][adapter_layer].view(-1)
            sim = cos_sim(tensor1, tensor2)
            sim_list.append(sim.item())
        weight_list.append(average(sim_list))

    weight_list = torch.tensor(weight_list)
    softmax_weight_list = torch.softmax(weight_list / temperature, 0)
    softmax_weight_list = softmax_weight_list.tolist()
    print(TASK_LIST[i])
    print(softmax_weight_list)

    unit_state_dict = OrderedDict()
    unit_state_dict['args'] = state_dict1['args']
    unit_state_dict['adapter'] = {}
    unit_state_dict['current_state'] = state_dict1['current_state']

    #unit_models
    for (k, v) in state_dict['adapter'].items():
        weight_sum = 0
        tensor_weight = v - v
        for num in range(len(new_state_list)):
            tensor_weight += new_state_list[num]['adapter'][
                k] * softmax_weight_list[num]
        unit_state_dict['adapter'][k] = tensor_weight

    torch.save(unit_state_dict, checkpoint_path)