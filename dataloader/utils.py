import os
import json
import re
import string
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler


#                   task_names_multi,
class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        # self.task_names_multi = task_names_multi
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        # 不用输入输出不一样长的数据集
        # assert len(self.input_ids) == len(self.decoder_input_ids)
        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], \
                self.decoder_input_ids[idx], self.decoder_attention_mask[idx]
        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        # 在函数调用中，* 能够将元组或列表解包成不同的参数
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        #之所以这样处理，是为了针对答案的数量不止一个的任务，该代码的作用是train的时候从许多答案中任选一个作为答案，而test的时候选择答案的第一个作为答案
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyQAMultiDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 task_names_multi=None,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.task_names_multi=task_names_multi
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        # 不用输入输出不一样长的数据集
        # assert len(self.input_ids) == len(self.decoder_input_ids)
        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], \
                self.decoder_input_ids[idx], self.decoder_attention_mask[idx], self.task_names_multi[idx]
        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], self.task_names_multi[idx]

class MyQAPromptDataset_AE(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 task_prefix, task_names, all_task_prompts, ontology, in_metadata=None, out_metadata=None,
                 is_training=False, prompt_num=100,
                 type1_num=25, type2_num=25,
                 general_num=50, task2id=None):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]
        
        # 每一行input_ids前面加上-1 ~ -100, 去掉后100列; attention_mask前面加上[1] * prompt_num
        input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in input_ids]
        self.input_ids = torch.LongTensor(input_ids)
        attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in attention_mask]
        self.attention_mask = torch.LongTensor(attention_mask)
        
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)

        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        self.all_task_prompts = all_task_prompts
        self.task_prefix = task_prefix
        self.task_names = task_names

        self.ontology = ontology
        self.type1_num = type1_num
        self.type2_num = type2_num
        self.general_num = general_num
        self.task2id = task2id

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]==len(self.task_prefix)==len(self.ontology)
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        # print('new')
        # print(self.ontology[idx])
        # print(self.task2id[self.task_names[idx]])
        ontology = self.ontology[idx]
        ontology_type1 = torch.LongTensor(list(range(ontology[0] * self.type1_num, (ontology[0] + 1) * self.type1_num)))
        ontology_type2 = torch.LongTensor(list(range(ontology[1] * self.type2_num, (ontology[1] + 1) * self.type2_num)))
        ontology_tensor_general = torch.LongTensor(list(range(self.task2id[self.task_names[idx]] * self.general_num, (self.task2id[self.task_names[idx]] + 1) * self.general_num)))

        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], random.choice(self.all_task_prompts[self.task_prefix[idx]]), ontology_type1, ontology_type2, ontology_tensor_general, self.task_prefix[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        task_prompt = random.choice(self.all_task_prompts[self.task_prefix[idx]])
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx], task_prompt, ontology_type1, ontology_type2, ontology_tensor_general

class MyQAPromptDataset_intrinsic(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 task_prefix, in_metadata=None, out_metadata=None,
                 is_training=False, prompt_num=100):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]
        
        # 每一行input_ids前面加上-1 ~ -100, 去掉后100列; attention_mask前面加上[1] * prompt_num
        input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in input_ids]
        self.input_ids = torch.LongTensor(input_ids)
        attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in attention_mask]
        self.attention_mask = torch.LongTensor(attention_mask)
        
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.task_prefix = task_prefix

        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        # print('new')
        # print(self.ontology[idx])
        # print(self.task2id[self.task_names[idx]])
        
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], self.task_prefix[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MyQAPromptDataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False, prompt_num=100):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]
        
        # 每一行input_ids前面加上-1 ~ -100, 去掉后100列; attention_mask前面加上[1] * prompt_num
        input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in input_ids]
        self.input_ids = torch.LongTensor(input_ids)
        attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in attention_mask]
        self.attention_mask = torch.LongTensor(attention_mask)
        
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))

        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyQAPromptBlendDataset(Dataset):
    def __init__(self, input_ids_blend,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False, prompt_num=100):
        self.input_ids_blend = torch.LongTensor(input_ids_blend)
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]
        
        # 每一行input_ids前面加上-1 ~ -100, 去掉后100列; attention_mask前面加上[1] * prompt_num
        input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in input_ids]
        self.input_ids = torch.LongTensor(input_ids)
        attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in attention_mask]
        self.attention_mask = torch.LongTensor(attention_mask)
        
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids_blend)==len(self.input_ids)
        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids_blend[idx], self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))

        return self.input_ids_blend[in_idx], self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class MyQAPromptBlend_blendTestDataset(Dataset):
    def __init__(self, input_ids_blend, in_metadata=None, is_training=False):
        self.input_ids_blend = torch.LongTensor(input_ids_blend)
        self.in_metadata = list(zip(range(len(input_ids_blend)), range(1, 1+len(input_ids_blend)))) \
            if in_metadata is None else in_metadata
        self.is_training = is_training

    def __len__(self):
        return len(self.input_ids_blend)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids_blend[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        return self.input_ids_blend[in_idx]


class MyDataLoader(DataLoader):
    def __init__(self, args, dataset, is_training):
        if is_training:
            if hasattr(args, 'local_rank') and args.local_rank > -1:
                sampler = DistributedSampler(dataset)
            else:
                sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


    # def __init__(self, args, dataset, is_training):
    #     if is_training:
    #         sampler=RandomSampler(dataset)
    #         batch_size = args.train_batch_size
    #     else:
    #         sampler=SequentialSampler(dataset)
    #         batch_size = args.predict_batch_size
    #     super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


class MyMetaLearningDataset(Dataset):
    def __init__(self,
                 train_input_ids, train_attention_mask, 
                 train_decoder_input_ids, train_decoder_attention_mask,
                 train_metadata_task, train_metadata_questions,
                 dev_input_ids, dev_attention_mask,
                 dev_decoder_input_ids, dev_decoder_attention_mask,
                 dev_metadata_task, dev_metadata_questions, 
                 inner_bsz,
                 is_training=False):

        self.train_input_ids = torch.LongTensor(train_input_ids)
        self.train_attention_mask = torch.LongTensor(train_attention_mask)

        self.train_decoder_input_ids = torch.LongTensor(train_decoder_input_ids)
        self.train_decoder_attention_mask = torch.LongTensor(train_decoder_attention_mask)

        self.dev_input_ids = torch.LongTensor(dev_input_ids)
        self.dev_attention_mask = torch.LongTensor(dev_attention_mask)

        self.dev_decoder_input_ids = torch.LongTensor(dev_decoder_input_ids)
        self.dev_decoder_attention_mask = torch.LongTensor(dev_decoder_attention_mask)

        self.train_metadata_task = train_metadata_task
        self.train_metadata_questions = train_metadata_questions

        self.dev_metadata_task = dev_metadata_task
        self.dev_metadata_questions = dev_metadata_questions

        self.inner_bsz = inner_bsz
        self.is_training = is_training

        assert len(self.train_input_ids)==len(self.train_attention_mask)==self.train_metadata_task[-1][-1]
        assert len(self.train_decoder_input_ids)==len(self.train_decoder_attention_mask)==self.train_metadata_questions[-1][-1]

        assert len(self.dev_input_ids)==len(self.dev_attention_mask)==self.dev_metadata_task[-1][-1]
        assert len(self.dev_decoder_input_ids)==len(self.dev_decoder_attention_mask)==self.dev_metadata_questions[-1][-1]

        assert len(self.train_metadata_task) == len(self.dev_metadata_task)

    def __len__(self):
        return len(self.train_metadata_task)

    def __getitem__(self, idx):
        # train
        if self.inner_bsz <= self.train_metadata_task[idx][1] - self.train_metadata_task[idx][0]:
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=True)

        train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask = [], [], [], []
        for train_in_index in train_in_indices:
            train_input_ids.append(self.train_input_ids[train_in_index])
            train_attention_mask.append(self.train_attention_mask[train_in_index])

            train_out_idx = np.random.choice(range(*self.train_metadata_questions[train_in_index]))

            train_decoder_input_ids.append(self.train_decoder_input_ids[train_out_idx])
            train_decoder_attention_mask.append(self.train_decoder_attention_mask[train_out_idx])

        train_input_ids = torch.stack(train_input_ids)
        train_attention_mask = torch.stack(train_attention_mask)
        train_decoder_input_ids = torch.stack(train_decoder_input_ids)
        train_decoder_attention_mask = torch.stack(train_decoder_attention_mask)

        # dev
        if self.inner_bsz <= self.dev_metadata_task[idx][1] - self.dev_metadata_task[idx][0]:
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=True)

        dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask = [], [], [], []
        for dev_in_index in dev_in_indices:
            dev_input_ids.append(self.dev_input_ids[dev_in_index])
            dev_attention_mask.append(self.dev_attention_mask[dev_in_index])

            dev_out_idx = np.random.choice(range(*self.dev_metadata_questions[dev_in_index]))

            dev_decoder_input_ids.append(self.dev_decoder_input_ids[dev_out_idx])
            dev_decoder_attention_mask.append(self.dev_decoder_attention_mask[dev_out_idx])

        dev_input_ids = torch.stack(dev_input_ids)
        dev_attention_mask = torch.stack(dev_attention_mask)
        dev_decoder_input_ids = torch.stack(dev_decoder_input_ids)
        dev_decoder_attention_mask = torch.stack(dev_decoder_attention_mask)

        return train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask, \
            dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask

class MyMetaLearningPromptDataset(Dataset):
    def __init__(self,
                 train_input_ids, train_attention_mask, 
                 train_decoder_input_ids, train_decoder_attention_mask,
                 train_metadata_task, train_metadata_questions,
                 dev_input_ids, dev_attention_mask,
                 dev_decoder_input_ids, dev_decoder_attention_mask,
                 dev_metadata_task, dev_metadata_questions, 
                 inner_bsz,
                 is_training=False, prompt_num=100):
        self.prompt_prefix = [- (i + 1) for i in range(prompt_num)]

        train_input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in train_input_ids]
        self.train_input_ids = torch.LongTensor(train_input_ids)
        train_attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in train_attention_mask]
        self.train_attention_mask = torch.LongTensor(train_attention_mask)

        self.train_decoder_input_ids = torch.LongTensor(train_decoder_input_ids)
        self.train_decoder_attention_mask = torch.LongTensor(train_decoder_attention_mask)

        dev_input_ids = [self.prompt_prefix + input_id[0:-(prompt_num)] for input_id in dev_input_ids]
        self.dev_input_ids = torch.LongTensor(dev_input_ids)
        dev_attention_mask = [[1]*prompt_num + atten_mask[0:-(prompt_num)] for atten_mask in dev_attention_mask]
        self.dev_attention_mask = torch.LongTensor(dev_attention_mask)

        self.dev_decoder_input_ids = torch.LongTensor(dev_decoder_input_ids)
        self.dev_decoder_attention_mask = torch.LongTensor(dev_decoder_attention_mask)

        self.train_metadata_task = train_metadata_task
        self.train_metadata_questions = train_metadata_questions

        self.dev_metadata_task = dev_metadata_task
        self.dev_metadata_questions = dev_metadata_questions

        self.inner_bsz = inner_bsz
        self.is_training = is_training

        assert len(self.train_input_ids)==len(self.train_attention_mask)==self.train_metadata_task[-1][-1]
        assert len(self.train_decoder_input_ids)==len(self.train_decoder_attention_mask)==self.train_metadata_questions[-1][-1]

        assert len(self.dev_input_ids)==len(self.dev_attention_mask)==self.dev_metadata_task[-1][-1]
        assert len(self.dev_decoder_input_ids)==len(self.dev_decoder_attention_mask)==self.dev_metadata_questions[-1][-1]

        assert len(self.train_metadata_task) == len(self.dev_metadata_task)

    def __len__(self):
        return len(self.train_metadata_task)

    def __getitem__(self, idx):
        # train
        if self.inner_bsz <= self.train_metadata_task[idx][1] - self.train_metadata_task[idx][0]:
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            train_in_indices = np.random.choice(range(*self.train_metadata_task[idx]), self.inner_bsz, replace=True)

        train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask = [], [], [], []
        for train_in_index in train_in_indices:
            train_input_ids.append(self.train_input_ids[train_in_index])
            train_attention_mask.append(self.train_attention_mask[train_in_index])

            train_out_idx = np.random.choice(range(*self.train_metadata_questions[train_in_index]))

            train_decoder_input_ids.append(self.train_decoder_input_ids[train_out_idx])
            train_decoder_attention_mask.append(self.train_decoder_attention_mask[train_out_idx])

        train_input_ids = torch.stack(train_input_ids)
        train_attention_mask = torch.stack(train_attention_mask)
        train_decoder_input_ids = torch.stack(train_decoder_input_ids)
        train_decoder_attention_mask = torch.stack(train_decoder_attention_mask)

        # dev
        if self.inner_bsz <= self.dev_metadata_task[idx][1] - self.dev_metadata_task[idx][0]:
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=False)
        else:
            # if there is not enough examples in the current task, we do `sample with replacement` to fill the batch
            dev_in_indices = np.random.choice(range(*self.dev_metadata_task[idx]), self.inner_bsz, replace=True)

        dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask = [], [], [], []
        for dev_in_index in dev_in_indices:
            dev_input_ids.append(self.dev_input_ids[dev_in_index])
            dev_attention_mask.append(self.dev_attention_mask[dev_in_index])

            dev_out_idx = np.random.choice(range(*self.dev_metadata_questions[dev_in_index]))

            dev_decoder_input_ids.append(self.dev_decoder_input_ids[dev_out_idx])
            dev_decoder_attention_mask.append(self.dev_decoder_attention_mask[dev_out_idx])

        dev_input_ids = torch.stack(dev_input_ids)
        dev_attention_mask = torch.stack(dev_attention_mask)
        dev_decoder_input_ids = torch.stack(dev_decoder_input_ids)
        dev_decoder_attention_mask = torch.stack(dev_decoder_attention_mask)

        return train_input_ids, train_attention_mask, train_decoder_input_ids, train_decoder_attention_mask, \
            dev_input_ids, dev_attention_mask, dev_decoder_input_ids, dev_decoder_attention_mask


class MyMetaLearningDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size

        super(MyMetaLearningDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)
        self.collate_fn = self.dummy_collate
        self.args = args

    def dummy_collate(self, input_data):
        return input_data

    def inference_dataloader(self):
        bsz = self.args.predict_batch_size
        for idx, (start_idx, end_idx) in enumerate(self.dataset.metadata_rel):
            input_ids_for_this_rel = self.dataset.input_ids[start_idx: end_idx]
            masks_for_this_rel = self.dataset.attention_mask[start_idx: end_idx]
            for j in range(0, len(input_ids_for_this_rel), bsz):
                input_ids_this_batch = input_ids_for_this_rel[j: j+bsz]
                masks_for_this_batch = masks_for_this_rel[j: j+bsz]

                yield self.dataset.relation_ids[idx], self.dataset.relation_mask[idx], input_ids_this_batch, masks_for_this_batch

class MyBlendDataset(Dataset):
    def __init__(self, input_all, data_type, sampling_size, sampling_num, 
                in_metadata=None, out_metadata=None,
                is_training=False):
        input_ids = []
        dataset_label = []
        for i, input_task_i in enumerate(input_all):
            # task_id = torch.tensor([int(i)])
            # label_task = [0] * len(input_all)
            if data_type == "train":
                for j in range(sampling_num):
                    input_ids.append(random.sample(input_task_i, sampling_size))
                    # label_task[i] = 1
                    # dataset_label.append(label_task)
                    dataset_label.append(i)
            elif data_type == "dev":
                # 32行转化为4份
                for j in range(int(32/sampling_size)):
                    input_ids.append(input_task_i[j*sampling_size : (j+1)*sampling_size])
                    dataset_label.append(i)

        self.input_ids = torch.LongTensor(input_ids)
        self.dataset_label = torch.LongTensor(dataset_label)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(dataset_label)), range(1, 1+len(dataset_label)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.dataset_label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # if not self.is_training:
        #     idx = self.in_metadata[idx][0]
        #     return self.input_ids[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[idx], self.dataset_label[idx]

class MyBlendDevDataset(Dataset):
    def __init__(self, dev_input, dev_label):

        self.dev_input = dev_input
        self.dev_label = dev_label

    def __len__(self):
        return len(self.dev_input)

    def __getitem__(self, idx):
        return self.dev_input[idx], self.dev_label[idx]
