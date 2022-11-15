import torch
import sys
from collections import OrderedDict
import copy
import sys
import os

sys.path.append('..')
from task_list import QA_task_list as TASK_LIST
from pro_hot_pic import pro_hot_pic

tem = 0.1
origin_data_path = '/home/lvxingtai/lxt/delta_search_code/result/adapter/Loss/result.tsv'
source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing'
save_ckpt_path = '/home/lvxingtai/lxt/delta_search_code/checkpoints/adapter/Loss'
os.makedirs(save_ckpt_path, exist_ok=True)

with open(origin_data_path, 'r') as fin:
    origin_data = fin.readlines()

state_list_full = []
for task in TASK_LIST:
    path = source_ckpt_path + '/' + task + '/checkpoint-best.pt'
    state_dict = torch.load(path)
    state_list_full.append(state_dict)

for i in range(len(state_list_full)):
    checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'
    new_state_list = copy.deepcopy(state_list_full)
    new_state_list.pop(i)

    #get_weight
    task_origin_data = origin_data[i * 33:i * 33 + 33]
    task_origin_data.pop(0)
    task_origin_data.pop(i)
    weight_list = []
    for data in task_origin_data:
        data = data.strip().split(",")
        weight_list.append(float(data[1]) * (-1))

    weight_list = torch.tensor(weight_list)
    softmax_weight_list = torch.softmax(weight_list / tem, 0)
    softmax_weight_list = softmax_weight_list.tolist()
    with open(save_ckpt_path + '/final_weight.txt','a') as fout:
        fout.write(TASK_LIST[i] + '\n')
        fout.write(str(softmax_weight_list) + '\n')

    unit_state_dict = OrderedDict()
    unit_state_dict['args'] = state_dict['args']
    unit_state_dict['adapter'] = {}
    unit_state_dict['current_state'] = state_dict['current_state']

    #unit_models
    for (k, v) in state_dict['adapter'].items():
        weight_sum = 0
        tensor_weight = v - v
        for num in range(len(new_state_list)):
            tensor_weight += new_state_list[num]['adapter'][
                k] * softmax_weight_list[num]
        unit_state_dict['adapter'][k] = tensor_weight

    torch.save(unit_state_dict, checkpoint_path)

data_info_file = save_ckpt_path + '/final_weight.txt'
ckpt_path = save_ckpt_path
pro_hot_pic(data_info_file, ckpt_path)