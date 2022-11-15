import numpy as np
import matplotlib.pyplot as plt

def pro_hot_pic(data_info_file, ckpt_path):
    #get data
    with open(data_info_file,'r') as fin:
        initial_data = fin.readlines()

    data_list = []
    for i in range(32):
        initial_data_list = initial_data[i*2+1].strip().strip('[').strip(']').split(',')
        task_data_list = []
        for data in initial_data_list:
            task_data_list.append(float(data.strip(' ')))
        task_data_list.insert(i,0.)
        data_list.extend(task_data_list)

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
    plt.savefig(ckpt_path+'/hot_pic_all.png')
    plt.clf()