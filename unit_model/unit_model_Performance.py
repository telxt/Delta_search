import torch
import sys
from collections import OrderedDict
import copy
import sys
import os

sys.path.append('..')
from task_list import QA_task_list as TASK_LIST
from pro_hot_pic import pro_hot_pic

tem = 0.001
source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing'
tuned_ckpt_path = '/home/lvxingtai/lxt/delta_search_code/result/adapter/qa_tasks/tune_200'
save_ckpt_path = '/home/lvxingtai/lxt/delta_search_code/checkpoints/adapter/Performance'
os.makedirs(save_ckpt_path, exist_ok=True)

with open(tuned_ckpt_path + '/zero_result_dev.tsv','r') as fin:
    zero_shot_data = fin.readlines()

for i in range(len(TASK_LIST)):
    checkpoint_path = save_ckpt_path + '/without_' + TASK_LIST[i] + '_checkpoint.pt'

    tune_result = []
    with open(tuned_ckpt_path + '/' + TASK_LIST[i] + '/result.tsv','r') as fin:
        tune_result_data = fin.readlines()
    for initial_data in tune_result_data:
        test_perf = initial_data.strip().split(",")
        if len(test_perf) > 1:
            tune_result.append(float(test_perf[1]))
    tune_result.pop(i)

    zeroshot_result = []
    for initial_data in zero_shot_data[i*3:i*3+3]:
        zeroshot_result.append(float(initial_data.strip()))
    zeroshot_result.pop(i)

    perf_gap = []
    for num in range(len(tune_result)):
        perf_gap.append(tune_result[num]-zeroshot_result[num])

    weight_list = torch.tensor(perf_gap)
    softmax_weight_list = torch.softmax(weight_list/tem, 0)
    softmax_weight_list = softmax_weight_list.tolist()
    with open(save_ckpt_path + '/final_weight.txt','a') as fout:
        fout.write(TASK_LIST[i] + '\n')
        fout.write(str(softmax_weight_list) + '\n')

    state_list_200 = []
    for task in TASK_LIST:
        path = tuned_ckpt_path + '/' + TASK_LIST[i]+'/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_200.append(state_dict)
    state_list_200.pop(i)

    unit_state_dict = OrderedDict()
    unit_state_dict['args'] = state_dict['args']
    unit_state_dict['adapter'] = {}
    unit_state_dict['current_state'] = state_dict['current_state']

    #unit_models
    for (k, v) in state_dict['adapter'].items():
        weight_sum = 0 
        tensor_weight = v - v
        for num in range(len(state_list_200)):
            tensor_weight += state_list_200[num]['adapter'][k]* softmax_weight_list[num]
        unit_state_dict['adapter'][k] = tensor_weight
    
    torch.save(unit_state_dict, checkpoint_path)

data_info_file = save_ckpt_path + '/final_weight.txt'
ckpt_path = save_ckpt_path
pro_hot_pic(data_info_file, ckpt_path)