from numpy import average
import torch
from numpy import average
from collections import OrderedDict
import copy
from task_list import QA_task_list as TASK_LIST
delta = 'adapter_softmax_64'

if delta == 'adapter':
    state_list = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    cos_sim = torch.nn.CosineSimilarity(dim=0)
    for i in range(len(state_list)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/cos_sim_tasks/without_'+TASK_LIST[i]+'_checkpoint.pt'

        #get_weight
        weight_list = []
        state_dict1 = state_list[i]
        for j in range(len(state_list)):
            state_dict2 = state_list[j]
            sim_list = []
            for adapter_layer in state_dict1['adapter']:
                tensor1 = state_dict1['adapter'][adapter_layer].view(-1)
                tensor2 = state_dict2['adapter'][adapter_layer].view(-1)
                sim = cos_sim(tensor1,tensor2)
                sim_list.append(sim.item())
            weight_list.append(average(sim_list))

        unit_state_dict = OrderedDict()
        unit_state_dict['args'] = state_dict1['args']
        unit_state_dict['adapter'] = {}
        unit_state_dict['current_state'] = state_dict1['current_state']
        #unit_models
        for (k, v) in state_dict['adapter'].items():
            weight_sum = 0 
            tensor_weight = v - v
            for num in range(len(state_list)):
                if not (num==i):
                    tensor_weight += state_list[num]['adapter'][k]* weight_list[num]
                    weight_sum += weight_list[num]
            tensor_weight /= weight_sum
            unit_state_dict['adapter'][k] = tensor_weight
        
        torch.save(unit_state_dict, checkpoint_path)

if delta == 'lora':
    state_list = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_lora/'+task\
            +'-lora_size_10-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    cos_sim = torch.nn.CosineSimilarity(dim=0)
    for i in range(len(state_list)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/lora/qa_tasks/cos_sim_tasks/without_'+TASK_LIST[i]+'_checkpoint.pt'

        #get_weight
        weight_list = []
        state_dict1 = state_list[i]
        for j in range(len(state_list)):
            state_dict2 = state_list[j]
            sim_list = []
            for lora_layer in state_dict1['lora']:
                tensor1 = state_dict1['lora'][lora_layer].view(-1)
                tensor2 = state_dict2['lora'][lora_layer].view(-1)
                sim = cos_sim(tensor1,tensor2)
                sim_list.append(sim.item())
            weight_list.append(average(sim_list))

        unit_state_dict = OrderedDict()
        unit_state_dict['args'] = state_dict1['args']
        unit_state_dict['lora'] = {}
        unit_state_dict['current_state'] = state_dict1['current_state']
        #unit_models
        for (k, v) in state_dict['lora'].items():
            weight_sum = 0 
            tensor_weight = v - v
            for num in range(len(state_list)):
                if not (num==i):
                    tensor_weight += state_list[num]['lora'][k]* weight_list[num]
                    weight_sum += weight_list[num]
            tensor_weight /= weight_sum
            unit_state_dict['lora'][k] = tensor_weight
        
        torch.save(unit_state_dict, checkpoint_path)

if delta == '2333':
    state_list = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_lora/'+task\
            +'-lora_size_10-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    cos_sim = torch.nn.CosineSimilarity(dim=0)
    with open('cos_sim_result.tsv','a') as fout:
        for i in range(len(state_list)):
            fout.write(TASK_LIST[i]+'\n')

            #get_weight
            state_dict1 = state_list[i]
            for j in range(len(state_list)):
                state_dict2 = state_list[j]
                sim_list = []
                for lora_layer in state_dict1['lora']:
                    tensor1 = state_dict1['lora'][lora_layer].view(-1)
                    tensor2 = state_dict2['lora'][lora_layer].view(-1)
                    sim = cos_sim(tensor1,tensor2)
                    sim_list.append(sim.item())
                
                fout.write(TASK_LIST[j]+','+str(average(sim_list))+'\n')

if delta == 'adapter_softmax':
    state_list = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list.append(state_dict)

    cos_sim = torch.nn.CosineSimilarity(dim=0)
    for i in range(len(state_list)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/cos_sim_tasks_softmax/without_'+TASK_LIST[i]+'_checkpoint.pt'

        #get_weight
        weight_list = []
        state_dict1 = state_list[i]

        new_state_list = copy.deepcopy(state_list)
        new_state_list.pop(i)

        for j in range(len(new_state_list)):
            state_dict2 = new_state_list[j]
            sim_list = []
            for adapter_layer in state_dict1['adapter']:
                tensor1 = state_dict1['adapter'][adapter_layer].view(-1)
                tensor2 = state_dict2['adapter'][adapter_layer].view(-1)
                sim = cos_sim(tensor1,tensor2)
                sim_list.append(sim.item())
            weight_list.append(average(sim_list))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list, 0)
        softmax_weight_list = softmax_weight_list.tolist()

        unit_state_dict = OrderedDict()
        unit_state_dict['args'] = state_dict1['args']
        unit_state_dict['adapter'] = {}
        unit_state_dict['current_state'] = state_dict1['current_state']

        #unit_models
        for (k, v) in state_dict['adapter'].items():
            weight_sum = 0 
            tensor_weight = v - v
            for num in range(len(new_state_list)):
                tensor_weight += new_state_list[num]['adapter'][k]* softmax_weight_list[num]
            unit_state_dict['adapter'][k] = tensor_weight
        
        torch.save(unit_state_dict, checkpoint_path)

if delta == 'adapter_64tune':
    state_list_64 = []
    for task in TASK_LIST:
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/all_tasks/64shot_random/'+task+'_best.pt'
        state_dict = torch.load(path)
        state_list_64.append(state_dict)

    state_list_full = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

        cos_sim = torch.nn.CosineSimilarity(dim=0)
        for i in range(len(state_list_64)):
            checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/cos_sim_tasks_64tune/without_'+TASK_LIST[i]+'_checkpoint.pt'

            fout.write(TASK_LIST[i]+'\n')
            
            #get_weight
            weight_list = []
            state_dict1 = state_list_64[i]
            for j in range(len(state_list_full)):
                state_dict2 = state_list_full[j]
                sim_list = []
                for adapter_layer in state_dict1['adapter']:
                    tensor1 = state_dict1['adapter'][adapter_layer].view(-1)
                    tensor2 = state_dict2['adapter'][adapter_layer].view(-1)
                    sim = cos_sim(tensor1,tensor2)
                    sim_list.append(sim.item())
                weight_list.append(average(sim_list))

                fout.write(TASK_LIST[j]+','+str(average(sim_list))+'\n')

            unit_state_dict = OrderedDict()
            unit_state_dict['args'] = state_dict1['args']
            unit_state_dict['adapter'] = {}
            unit_state_dict['current_state'] = state_dict1['current_state']
            #unit_models
            for (k, v) in state_dict['adapter'].items():
                weight_sum = 0 
                tensor_weight = v - v
                for num in range(len(state_list_full)):
                    if not (num==i):
                        tensor_weight += state_list_full[num]['adapter'][k]* weight_list[num]
                        weight_sum += weight_list[num]
                        print(num)
                tensor_weight /= weight_sum
                unit_state_dict['adapter'][k] = tensor_weight
            
            torch.save(unit_state_dict, checkpoint_path)

if delta == 'adapter_softmax_64':
    temperature = 0.001
    state_list_64 = []
    for task in TASK_LIST:
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/random/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_64.append(state_dict)

    state_list_full = []
    for task in TASK_LIST:
        # path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
        #         +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    cos_sim = torch.nn.CosineSimilarity(dim=0)
    for i in range(len(state_list_64)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/cossim_tem0.001/without_'+TASK_LIST[i]+'_checkpoint.pt'

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
                sim = cos_sim(tensor1,tensor2)
                sim_list.append(sim.item())
            weight_list.append(average(sim_list))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/temperature, 0)
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
                tensor_weight += new_state_list[num]['adapter'][k]* softmax_weight_list[num]
            unit_state_dict['adapter'][k] = tensor_weight
        
        torch.save(unit_state_dict, checkpoint_path)