import torch
import sys
from collections import OrderedDict
import copy
from task_list import QA_task_list as TASK_LIST
mode = 'perf_change_all_task'


if mode == 'data_similar':
    tem = 0.1
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/loss_tem0.1/initial_data.tsv','r') as fin:
        loss_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        # path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
        #         +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/loss_tem0.1/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_loss_data = loss_data[i*33:i*33+33]
        task_loss_data.pop(0)
        task_loss_data.pop(i)
        weight_list = []
        for data in task_loss_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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

if mode == 'cum_weight_gap':
    for task in TASK_LIST:
        #load_state
        initial_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        tune_path = '/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/perf_change/' \
            +'get_perf_and_ckpt/superglue-record/'+task+'/checkpoint-best.pt'

        initial_state = torch.load(initial_path)
        tune_state = torch.load(tune_path)

        #cum_weight_gap
        gap_sum = 0
        for (k, v) in initial_state['adapter'].items():
            weight_gap = v - tune_state['adapter'][k]
            gap_dis = torch.norm(input=weight_gap)
            gap_sum += float(gap_dis)

        with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/perf_change/get_perf_and_ckpt/superglue-record/weight_gap.tsv','a')\
            as fout:
            fout.write(task+','+str(gap_sum)+'\n')

if mode == 'weight_gap':
    #!很可能有问题
    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/perf_change/get_perf_and_ckpt/superglue-record/weight_gap.tsv','r') as fin:
        gap_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/data_similar10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_loss_data = loss_data[i*33:i*33+33]
        task_loss_data.pop(0)
        task_loss_data.pop(i)
        weight_list = []
        for data in task_loss_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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
     
if mode == 'weight_gap_all_task':
    tem = 0.1

    state_list_full = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(TASK_LIST)):
        #cum weight_gap
        weight_gap_list = []

        for j in range(len(TASK_LIST)):
            if j != i:
                #load_state
                initial_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+TASK_LIST[j] \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                tune_path = '/data/private/lvxingtai/delta_search_result/adapter_qa_perfchange_get200ckpt/'+TASK_LIST[i]+'/'+TASK_LIST[j]+'/checkpoint-best.pt'

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
        print(TASK_LIST[i])
        print(softmax_weight_list)

        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/weight_gap_200_tem0.1/without_'+TASK_LIST[i]+'_checkpoint.pt'
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

if mode == 'weight_gap_alltask_aftertuneckpt':
    tem = 10
    for i in range(len(TASK_LIST)):
        #cum weight_gap
        weight_gap_list = []

        for j in range(len(TASK_LIST)):
            if j != i:
                #load_state
                initial_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+TASK_LIST[j] \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                tune_path = '/data/private/lvxingtai/delta_search_result/adapter_qa_perfchange_get200ckpt/'+TASK_LIST[i]+'/'+TASK_LIST[j]+'/checkpoint-best.pt'

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
        print(TASK_LIST[i])
        print(softmax_weight_list)

        
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/weight_gap_200_tunedckpt_tem10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        state_list_200 = []
        for task in TASK_LIST:
            path = '/data/private/lvxingtai/delta_search_result/adapter_qa_perfchange_get200ckpt/'+TASK_LIST[i]+'/'+task+'/checkpoint-best.pt'
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

if mode == 'perf_change_all_task':
    tem = 0.001
    with open('/data/private/lvxingtai/delta_search_result/adapter_qa_perfchange_get200ckpt/zero_result_dev.tsv','r') as fin:
        zero_shot_data = fin.readlines()

    for i in range(len(TASK_LIST)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/perfgap_fu_tem0.001/without_'+TASK_LIST[i]+'_checkpoint.pt'

        tune_result = []
        with open('/data/private/lvxingtai/delta_search_result/adapter_qa_perfchange_get200ckpt/'+TASK_LIST[i]+'/result.tsv','r') as fin:
            tune_result_data = fin.readlines()
        for initial_data in tune_result_data:
            test_perf = initial_data.strip().split(",")
            if len(test_perf) > 1:
                tune_result.append(float(test_perf[1]))
        tune_result.pop(i)

        zeroshot_result = []
        for initial_data in zero_shot_data[i*32:i*32+32]:
            zeroshot_result.append(float(initial_data.strip()))
        zeroshot_result.pop(i)

        perf_gap = []
        for num in range(len(tune_result)):
            perf_gap.append(tune_result[num]-zeroshot_result[num])

        weight_list = torch.tensor(perf_gap)
        softmax_weight_list = torch.softmax((-1)*weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

        state_list_200 = []
        for task in TASK_LIST:
            path = '/data/private/lvxingtai/delta_search_result/adapter_qa_perfchange_get200ckpt/'+TASK_LIST[i]+'/'+task+'/checkpoint-best.pt'
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

if mode == 'KL':
    tem = 10
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/KL_tem10/initial_data.tsv','r') as fin:
        KL_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        # path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
        #         +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/KL_tem10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_KL_data = KL_data[i*33:i*33+33]
        task_KL_data.pop(0)
        task_KL_data.pop(i)
        weight_list = []
        for data in task_KL_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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

if mode == 'EL2N':
    tem = 10
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/EL2N_tem1/initial_data.tsv','r') as fin:
        EL2N_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        # path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
        #         +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/EL2N_tem10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_EL2N_data = EL2N_data[i*33:i*33+33]
        task_EL2N_data.pop(0)
        task_EL2N_data.pop(i)
        weight_list = []
        for data in task_EL2N_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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

if mode == 'GraNd':
    tem = 10
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_large/GraNd_tem1/initial_data.tsv','r') as fin:
        EL2N_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        # path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
        #         +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_large/fulldata/'+task \
            +'-adapter_size_12-seed_44_t5_large/lr_0.0005_bsz_16_seed_44/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_large/GraNd_tem10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_EL2N_data = EL2N_data[i*33:i*33+33]
        task_EL2N_data.pop(0)
        task_EL2N_data.pop(i)
        weight_list = []
        for data in task_EL2N_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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

if mode == 'KL_encoder':
    tem = 0.01
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL_encoder_new_tem0.1/KL_encoder_new_initial_data.tsv','r') as fin:
        KL_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL_encoder_new_tem0.01/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_KL_data = KL_data[i*33:i*33+33]
        task_KL_data.pop(0)
        task_KL_data.pop(i)
        weight_list = []
        for data in task_KL_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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

if mode == 'GraNd_29':
    from task_list import just_task as TASK_LIST_new
    tem = 10
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_withouteli_tem10/GraNd_initial_data.tsv','r') as fin:
        EL2N_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST_new:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_withouteli_tem10/without_'+TASK_LIST_new[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_EL2N_data = EL2N_data[i*30:i*30+30]
        task_EL2N_data.pop(0)
        task_EL2N_data.pop(i)
        weight_list = []
        for data in task_EL2N_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST_new[i])
        print(softmax_weight_list)

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

if mode == 'logits_label_cos':
    tem = 10
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_2048shot/logits_label_cos_tem10/initial_data.tsv','r') as fin:
        EL2N_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        # path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
        #         +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_2048shot/logits_label_cos_tem10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_EL2N_data = EL2N_data[i*33:i*33+33]
        task_EL2N_data.pop(0)
        task_EL2N_data.pop(i)
        weight_list = []
        for data in task_EL2N_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1]))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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

if mode == 'GraNd_block':
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    tem = 0.1

    os.makedirs('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem), exist_ok=True)
    state_list_full = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    initial_data_list = [[] for i in range(24)]
    for i in range(12):
        with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/GraNd_block_initial_data/encoder_block_'+str(i)+'_result.tsv' ,'r') as fin:
            initial_data_list[i] = fin.readlines()
        os.makedirs('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/encoder_block_'+str(i), exist_ok=True)
        with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/GraNd_block_initial_data/decoder_block_'+str(i)+'_result.tsv' ,'r') as fin:
            initial_data_list[i+12] = fin.readlines()
        os.makedirs('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/decoder_block_'+str(i), exist_ok=True)

    for i in range(24):
        if i < 12:
            pic_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/encoder_block_'+str(i)+'/hotpic.png'
            info_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/encoder_block_'+str(i)+'/info.txt'
            ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/encoder_block_'+str(i)
        else:
            pic_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/decoder_block_'+str(i-12)+'/hotpic.png'
            info_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/decoder_block_'+str(i-12)+'/info.txt'
            ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/GraNd_block/tem_'+str(tem)+'/decoder_block_'+str(i-12)

        
        data_list = []

        for j in range(32):
            checkpoint_path = ckpt_path+'/without_'+TASK_LIST[j]+'_checkpoint.pt'
            new_state_list = copy.deepcopy(state_list_full)
            new_state_list.pop(j)

            task_data = initial_data_list[i][j*33:j*33+33]
            task = task_data[0].strip()
            task_data.pop(0)
            task_data.pop(j)

            weight_list = []
            for data in task_data:
                data = data.strip().split(",")
                weight_list.append(float(data[1])*(-1))
            weight_list = torch.tensor(weight_list)
            softmax_weight_list = torch.softmax(weight_list/tem, 0)
            softmax_weight_list = softmax_weight_list.tolist()

            with open(info_path, 'a') as fout:
                fout.write(task+'\n')
                fout.write(str(softmax_weight_list)+'\n')
            
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
                        
            softmax_weight_list.insert(j,0.)
            data_list.extend(softmax_weight_list)

        result = np.array(data_list)
        result.resize((32,32))

        QA_task_list = [
            'adversarialqa',
            'hotpot_qa',
            'superglue-record',

            'ai2_arc',
            'codah',
            'commonsense_qa',
            'cosmos_qa',
            'dream',
            'hellaswag',
            'openbookqa',
            'qasc',
            'quail',
            'quarel',
            'quartz-no_knowledge',
            'quartz-with_knowledge',
            'race-high',
            'race-middle',
            'sciq',
            'superglue-copa',
            'swag',
            'wino_grande',
            'wiqa',

            'boolq',
            'mc_taco',

            'eli5-askh',
            'eli5-asks',
            'eli5-eli5',
            
            'lama-conceptnet',
            'lama-google_re',
            'numer_sense',
            'search_qa',
            'web_questions',
        ]

        fig, ax = plt.subplots(dpi=600)
        plt.subplots_adjust(top=0.99, bottom=0.24, left=0.24, right=0.99)
        ax.set_xticks(np.arange(0, 32, 1))
        ax.set_xticklabels(QA_task_list,size=5)
        plt.setp(ax.get_xticklabels(), rotation=80,ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(0, 32, 1))
        ax.set_yticklabels(QA_task_list,size=5)
        plt.imshow(result, cmap='coolwarm', origin='upper', aspect='auto')
        plt.colorbar()
        plt.xlabel('ckpt',{'size':7})
        plt.ylabel('task',{'size':7})
        plt.savefig(pic_path)
        plt.clf()

if mode == 'block_last':
    tem = 10
    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/blockdecoder_dif_last_tem100/initial_data.tsv','r') as fin:
        EL2N_data = fin.readlines()

    state_list_full = []
    for task in TASK_LIST:
        path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
        state_dict = torch.load(path)
        state_list_full.append(state_dict)

    for i in range(len(state_list_full)):
        checkpoint_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/blockdecoder_dif_last_tem10/without_'+TASK_LIST[i]+'_checkpoint.pt'
        new_state_list = copy.deepcopy(state_list_full)
        new_state_list.pop(i)

        #get_weight
        task_EL2N_data = EL2N_data[i*33:i*33+33]
        task_EL2N_data.pop(0)
        task_EL2N_data.pop(i)
        weight_list = []
        for data in task_EL2N_data:
            data = data.strip().split(",")
            weight_list.append(float(data[1])*(-1))

        weight_list = torch.tensor(weight_list)
        softmax_weight_list = torch.softmax(weight_list/tem, 0)
        softmax_weight_list = softmax_weight_list.tolist()
        print(TASK_LIST[i])
        print(softmax_weight_list)

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
