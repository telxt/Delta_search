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
source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing'
tuned_ckpt_path = '/home/lvxingtai/lxt/delta_search_code/result/adapter/qa_tasks/tune_200'
save_ckpt_path = '/home/lvxingtai/lxt/delta_search_code/checkpoints/adapter/Euclidean'
os.makedirs(save_ckpt_path, exist_ok=True)


state_list_full = []
for task in TASK_LIST:
    path = source_ckpt_path + '/' + task + '/checkpoint-best.pt'
    state_dict = torch.load(path)
    state_list_full.append(state_dict)

for i in range(len(TASK_LIST)):
    #cum weight_gap
    weight_gap_list = []

    for j in range(len(TASK_LIST)):
        if j != i:
            #load_state
            initial_path = source_ckpt_path + '/' + TASK_LIST[j] + '/checkpoint-best.pt'
            tune_path = tuned_ckpt_path + '/'+TASK_LIST[i]+'/'+TASK_LIST[j]+'/checkpoint-best.pt'

            initial_state = torch.load(initial_path)
            tune_state = torch.load(tune_path)

            gap_sum = 0
            for (k, v) in initial_state['adapter'].items():
                weight_gap = v - tune_state['adapter'][k]
                gap_dis = torch.norm(input=weight_gap)
                gap_sum += float(gap_dis)

            weight_gap_list.append(gap_sum)

    weight_list = torch.tensor(weight_gap_list)
    softmax_weight_list = torch.softmax((-1)*weight_list/tem, 0)
    softmax_weight_list = softmax_weight_list.tolist()
    with open(save_ckpt_path + '/final_weight.txt','a') as fout:
        fout.write(TASK_LIST[i] + '\n')
        fout.write(str(softmax_weight_list) + '\n')

    checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'
    new_state_list = copy.deepcopy(state_list_full)
    new_state_list.pop(i)

    unit_state_dict = OrderedDict()
    unit_state_dict['args'] = state_dict['args']
    unit_state_dict['adapter'] = {}
    unit_state_dict['current_state'] = state_dict['current_state']

    #unit_models
    for (k, v) in state_dict['adapter'].items():
        weight_sum = 0 
        tensor_weight = v - v
        for num in range(len(new_state_list)):
            tensor_weight += new_state_list[num]['adapter'][k]* softmax_weight_list[num]
        unit_state_dict['adapter'][k] = tensor_weight
    
    torch.save(unit_state_dict, checkpoint_path)

data_info_file = save_ckpt_path + '/final_weight.txt'
ckpt_path = save_ckpt_path
pro_hot_pic(data_info_file, ckpt_path)