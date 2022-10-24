from hashlib import algorithms_available
import os
import numpy as np
import torch
import logging
import random
import math
import warnings
# import loralib as lora

# import lora_onlyB

from transformers import AutoTokenizer, BartTokenizer, BartConfig
from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
    get_linear_schedule_with_warmup,
    is_torch_available,
)
# from crossfit_yijing.module.bias_mlp import Embedding
from dataloader.fewshot_gym_singletask_t5 import NLPFewshotGymSingleTaskData
# from dataloader.fewshot_gym_singletask_t5_noextraid import NLPFewshotGymSingleTaskData

from transformers import T5ForConditionalGeneration
# from modelForGeneration import MyBart, MyT5
# from modelForGenerationPrompt import MyBartPrompt, MyT5Prompt
from intrinsic import intrinsic_dimension, intrinsic_dimension_said
from utils import freeze_embeds, trim_batch

from tqdm import tqdm
from collections import OrderedDict
import itertools
from torch.utils.tensorboard import SummaryWriter

# logger = logging.getLogger('trainer')
warnings.filterwarnings("ignore")

def uniform_init(prompt, a=0.0, b=1.0):
    torch.nn.init.uniform_(prompt, a, b)
    # logger.info("init prompt by uniform [{:.3f}, {:.3f}]".format(a, b))

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def get_params_for_prompt_optimization(module: torch.nn.Module):
    params = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

    # if torch.distributed.get_rank() == 0:
    #     print("print params", params)
    return params

class Trainer:
    def __init__(self, args, logger, model_provider):
        self.args = args
        self.logger = logger
        logger.info("Loading model ...")
        
        self.model, self.config, self.tokenizer = model_provider(args)
        if self.args.tune_method == 'fastfood':
            self.model, self.ID_wrap = intrinsic_dimension(self.model, args.intrinsic_dim, None, set(), args.projection_type, "cuda")
        
        # logger.info(self.model)
        logger.info("Loading Dataset ...")
        self.train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
        self.train_data.load_dataset(self.tokenizer)
        self.train_data.load_dataloader()
        self.dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)
        # self.dev_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=False)
        self.dev_data.load_dataset(self.tokenizer)
        self.dev_data.load_dataloader()
        self.test_data = NLPFewshotGymSingleTaskData(logger, args, args.test_file, data_type="test", is_training=False)
        self.test_data.load_dataset(self.tokenizer)
        self.test_data.load_dataloader()

        # self.train_dataset, self.valid_dataset, self.test_dataset = dataset_provider(args, self.tokenizer)
        self.device = self.init_device(args)
        self.model = self.model.to(self.device)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        # self.init_tensorboard(args)
        
        if args.seed is not None:
            set_seed(args.seed)
        if args.tune_method == 'prompt':
            self.prompt = torch.rand((args.prompt_num, self.config.d_model), requires_grad=True, device=self.device)
            self.prepare_data = self.prepare_prompt_data
            uniform_init(prompt=self.prompt, a=-math.sqrt(1 / self.config.d_model), b=math.sqrt(1 / self.config.d_model))
        elif args.tune_method == 'model' or args.tune_method == 'fastfood':
            self.prepare_data = self.prepare_model_data
        elif args.tune_method == 'lora' or args.tune_method == 'bias' or args.tune_method == 'prefix' or args.tune_method == 'adapter' or args.tune_method == 'hyper_PET':
             #和fine-tuning的输入一样？
            self.prepare_data = self.prepare_model_data
        elif args.tune_method == 'lora_stage2' or 'bias_stage2':
             #和fine-tuning的输入一样？
            self.prepare_data = self.prepare_model_data
        
        
        if args.tune_method == 'adapter' and args.SGD_noise:
            adapter_seed_42_path = args.adapter_init_seed_42_path
            state_dict = torch.load(adapter_seed_42_path)
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict.items()})
            self.model.load_state_dict(model_dict)
            logger.info(f'To keep the same adapter parameters as seed 42, loaded them from {adapter_seed_42_path}')
        elif args.tune_method == 'lora' and args.SGD_noise:
            lora_seed_42_path = args.lora_init_seed_42_path
            state_dict = torch.load(lora_seed_42_path)
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict.items()})
            self.model.load_state_dict(model_dict)
            logger.info(f'To keep the same lora parameters as seed 42, loaded them from {lora_seed_42_path}')
        elif args.tune_method == 'prefix' and args.SGD_noise:
            prefix_seed_42_path = args.prefix_init_seed_42_path
            state_dict = torch.load(prefix_seed_42_path)
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict.items()})
            self.model.load_state_dict(model_dict)
            logger.info(f'To keep the same prefix parameters as seed 42, loaded them from {prefix_seed_42_path}')

        if args.tune_method == 'lora_stage2' and not args.load_random_B:
            self.load_lora_B(args.load_lora_B_path)
        elif args.tune_method == 'bias_stage2':
            self.load_bias(args.load_bias_path)
        
    def init_device(self, args):
        if (not torch.cuda.is_available()):
            print('no gpu can be used!')
            assert torch.cuda.is_available()
        else:
            return torch.device('cuda:0')
    
    def init_tensorboard(self, args):
        self.tensorboard = None
        # if args.tensorboard_dir is not None:
        
        args.tensorboard_dir = args.output_dir + '/tensorboard'
        self.tensorboard = SummaryWriter(log_dir=args.tensorboard_dir)
    def get_optimzied_group(self):
        # 在这里frozen了（使用delta）时不需要tune的weight
        if self.args.tune_method == 'model':
            no_decay = ["bias", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            to_update = self.model.parameters()
        elif self.args.tune_method == 'fastfood':
            for n, p in self.model.named_parameters():
                if p.requires_grad == True:
                    print(n)
            optimizer_grouped_parameters = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad == True], 'weight_decay': 0.0}]
            to_update = self.model.parameters()
        elif self.args.tune_method == 'prompt':
            for n, p in self.model.named_parameters():
                p.requires_grad = False
            optimizer_grouped_parameters = [
                {
                    "params": [self.prompt],
                    "weight_decay": self.args.weight_decay,
                }
            ]
            to_update = [self.prompt]
        elif self.args.tune_method == 'lora':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
        
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            '''
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            '''
            to_update = self.model.parameters()
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum+=param.numel()
            print(sum)
        elif self.args.tune_method == 'adapter':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "adapter" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "adapter" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum+=param.numel()
            print(sum)
        elif self.args.tune_method == 'prefix':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "prefix" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "prefix" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum+=param.numel()
            print(sum)
            to_update = self.model.parameters()
        elif self.args.tune_method == 'hyper_PET':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "intrinsic" in n or 'hyper' in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "intrinsic" in n or 'hyper' in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        
        elif self.args.tune_method == 'bias_stage2' or  self.args.tune_method =='lora_stage2':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "lora_R" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora_R" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'bias':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                # if 'lora'
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
                    # optimizer_grouped_parameters.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        return optimizer_grouped_parameters, to_update

    def train(self):
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)

        if self.args.train_checkpoint:
            self.load_checkpoint(self.args.train_checkpoint)
            self.logger.info('load checkpoints from'+self.args.train_checkpoint)

        if self.args.tune_method == 'model' or self.args.tune_method == 'fastfood' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
            self.model.train()
        elif self.args.tune_method == 'prompt':
            self.model.eval()
        train_dataloader = self.train_data.dataloader
        if self.args.train_iters is None:
            self.args.train_iters = (
                                    len(train_dataloader) // self.gradient_accumulation_steps
                                    * float(self.args.train_epochs)
                                )
        if self.args.train_epochs is None:
            self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                                     // len(train_dataloader) + 1
        # if self.args.lr_decay_iters is None:
        #     self.args.lr_decay_iters = self.args.train_iters
        # setup optimizer
        optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
        # print(optimizer_grouped_parameters)
        self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
        warm_up_steps = self.args.warmup_steps
        self.scheduler = get_scheduler('constant', self.optimizer, warm_up_steps, self.args.train_iters)
        # self.scheduler =  get_linear_schedule_with_warmup(self.optimizer,
        #                                 num_warmup_steps=warm_up_steps,
        #                                 num_training_steps=self.args.train_iters)
        
        num_updates = 0
        log_dict = OrderedDict()
        best_metric = 0
        best_metric_dict = None
        best_num_updates = 0
        early_stop = 0

        # best_ckpt_path = f"{self.args.output_dir}/checkpoint-best.pt"
        if os.path.exists(self.args.output_dir):
            epoch, num_updates, best_metric = self.load_from_nearest(self.args.output_dir)
            best_num_updates = num_updates
        
        self.logger.info(f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")
        for epoch in range(self.args.train_epochs):
            self.optimizer.zero_grad()
            self.reset_logging(log_dict)
            
            for local_step, batch in enumerate(train_dataloader):
                loss = self.train_step(batch)
                self.add_logging(log_dict, 'loss', loss.item() * self.gradient_accumulation_steps)
                if local_step % self.gradient_accumulation_steps == 0:
                    # update model parameter 
                    # to_update_parameters
                    updated, old_scale = self.optimizer_step(self.model.parameters())
                    # updated, old_scale = self.optimizer_step(to_update_parameters)
                    if updated:
                        num_updates += 1
                    else:
                        self.logger.info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"\
                                    .format(old_scale, self.scaler.get_scale()))
                    if num_updates % self.args.log_interval == 0:
                        # to log
                        train_loss_mean = self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                      lr=self.scheduler.get_last_lr()[0])
                    self.reset_logging(log_dict)
                    if self.args.valid_interval is not None and \
                            num_updates % self.args.valid_interval == 0:
                        current_metrics = self.valid(epoch, num_updates)
                        best_update, average_score = self.early_stop(current_metrics, best_metric, epoch, num_updates)
                        if not best_update or train_loss_mean < 1e-7:
                            early_stop += 1
                            self.logger.info(f"Early stop + 1 = {early_stop}. " \
                                        f"Best averate score = {best_metric} at {best_num_updates}.")
                        else:
                            early_stop = 0
                            best_metric = average_score
                            best_num_updates = num_updates
                        if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                            break
                    if self.args.output_interval is not None and \
                            num_updates % self.args.output_interval == 0:
                        save_path = f"{self.args.output_dir}/checkpoint@{epoch}-{num_updates}.pt"
                        self.save_checkpoint(save_path, epoch, num_updates)
                        
                    if num_updates >= self.args.train_iters:
                        break
            
            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                self.logger.info(f"Stop training. Best averate score = {best_metric:.3f} at {best_num_updates}.")
                break
            if num_updates >= self.args.train_iters:
                break
        # if self.args.tune_method != 'model':
        #     save_path = f"{self.args.output_dir}/checkpoint-last.pt"
        #     self.save_checkpoint(save_path, epoch, num_updates)
        # if True:
        #     save_path = f"{self.args.output_dir}/checkpoint-last.pt"
        #     self.save_checkpoint(save_path, epoch, num_updates)

        return best_metric

    def train_multi(self):
        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)

        if self.args.train_checkpoint:
            self.load_checkpoint(self.args.train_checkpoint)
            self.logger.info('load checkpoints from'+self.args.train_checkpoint)

        if self.args.tune_method == 'model' or self.args.tune_method == 'fastfood' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
            self.model.train()
        elif self.args.tune_method == 'prompt':
            self.model.eval()
        train_dataloader = self.train_data.dataloader
        if self.args.train_iters is None:
            self.args.train_iters = (
                                    len(train_dataloader) // self.gradient_accumulation_steps
                                    * float(self.args.train_epochs)
                                )
        if self.args.train_epochs is None:
            self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                                     // len(train_dataloader) + 1
        # if self.args.lr_decay_iters is None:
        #     self.args.lr_decay_iters = self.args.train_iters
        # setup optimizer
        optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
        # print(optimizer_grouped_parameters)
        self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
        warm_up_steps = self.args.warmup_steps
        self.scheduler = get_scheduler('constant', self.optimizer, warm_up_steps, self.args.train_iters)
        # self.scheduler =  get_linear_schedule_with_warmup(self.optimizer,
        #                                 num_warmup_steps=warm_up_steps,
        #                                 num_training_steps=self.args.train_iters)
        
        num_updates = 0
        log_dict = OrderedDict()
        best_metric = 0
        best_metric_dict = None
        best_num_updates = 0
        early_stop = 0

        # best_ckpt_path = f"{self.args.output_dir}/checkpoint-best.pt"
        if os.path.exists(self.args.output_dir):
            epoch, num_updates, best_metric = self.load_from_nearest(self.args.output_dir)
            best_num_updates = num_updates
        
        self.logger.info(f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")
        for epoch in range(self.args.train_epochs):
            self.optimizer.zero_grad()
            self.reset_logging(log_dict)
            
            for local_step, batch in enumerate(train_dataloader):
                loss = self.train_step(batch)
                self.add_logging(log_dict, 'loss', loss.item() * self.gradient_accumulation_steps)
                if local_step % self.gradient_accumulation_steps == 0:
                    # update model parameter 
                    # to_update_parameters
                    updated, old_scale = self.optimizer_step(self.model.parameters())
                    # updated, old_scale = self.optimizer_step(to_update_parameters)
                    if updated:
                        num_updates += 1
                    else:
                        self.logger.info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"\
                                    .format(old_scale, self.scaler.get_scale()))
                    if num_updates % self.args.log_interval == 0:
                        # to log
                        train_loss_mean = self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                      lr=self.scheduler.get_last_lr()[0])
                    self.reset_logging(log_dict)
                        
                    if num_updates >= self.args.train_iters:
                        break
            
            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                self.logger.info(f"Stop training. Best averate score = {best_metric:.3f} at {best_num_updates}.")
                break
            if num_updates >= self.args.train_iters:
                break
        # if self.args.tune_method != 'model':
        #     save_path = f"{self.args.output_dir}/checkpoint-last.pt"
        #     self.save_checkpoint(save_path, epoch, num_updates)
        if True:
            save_path = f"{self.args.output_dir}/checkpoint-last.pt"
            self.save_checkpoint(save_path, epoch, num_updates)
    
    def train_and_test_on_all_ckpt(self, out_path, save_last):
        from task_list import QA_task_list as TASK_LIST

        for task in TASK_LIST:
            self.args.train_checkpoint = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
            self.args.test_checkpoint = None

            self.args.output_dir = out_path + '/' + task
            if not os.path.exists(self.args.output_dir):
                os.mkdir(self.args.output_dir)

            if self.args.train_checkpoint:
                self.load_checkpoint(self.args.train_checkpoint)
                self.logger.info('load checkpoints from'+self.args.train_checkpoint)

            if self.args.tune_method == 'model' or self.args.tune_method == 'fastfood' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
                self.model.train()
            elif self.args.tune_method == 'prompt':
                self.model.eval()
            train_dataloader = self.train_data.dataloader
            if self.args.train_iters is None:
                self.args.train_iters = (
                                        len(train_dataloader) // self.gradient_accumulation_steps
                                        * float(self.args.train_epochs)
                                    )
            if self.args.train_epochs is None:
                self.args.train_epochs = (self.args.train_iters * self.gradient_accumulation_steps) \
                                        // len(train_dataloader) + 1
            # if self.args.lr_decay_iters is None:
            #     self.args.lr_decay_iters = self.args.train_iters
            # setup optimizer
            optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
            # print(optimizer_grouped_parameters)
            self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
            warm_up_steps = self.args.warmup_steps
            self.scheduler = get_scheduler('constant', self.optimizer, warm_up_steps, self.args.train_iters)
            # self.scheduler =  get_linear_schedule_with_warmup(self.optimizer,
            #                                 num_warmup_steps=warm_up_steps,
            #                                 num_training_steps=self.args.train_iters)
            
            num_updates = 0
            log_dict = OrderedDict()
            best_metric = 0
            best_metric_dict = None
            best_num_updates = 0
            early_stop = 0

            # best_ckpt_path = f"{self.args.output_dir}/checkpoint-best.pt"
            if os.path.exists(self.args.output_dir):
                epoch, num_updates, best_metric = self.load_from_nearest(self.args.output_dir)
                best_num_updates = num_updates
            
            self.logger.info(f"Train {len(train_dataloader) // self.gradient_accumulation_steps} steps every epoch")
            for epoch in range(self.args.train_epochs):
                self.optimizer.zero_grad()
                self.reset_logging(log_dict)
                
                for local_step, batch in enumerate(train_dataloader):
                    loss = self.train_step(batch)
                    self.add_logging(log_dict, 'loss', loss.item() * self.gradient_accumulation_steps)
                    if local_step % self.gradient_accumulation_steps == 0:
                        # update model parameter 
                        # to_update_parameters
                        updated, old_scale = self.optimizer_step(self.model.parameters())
                        # updated, old_scale = self.optimizer_step(to_update_parameters)
                        if updated:
                            num_updates += 1
                        else:
                            self.logger.info("Inf or NaN detected in grad. Change scale from {:.1f} to {:.1f}"\
                                        .format(old_scale, self.scaler.get_scale()))
                        if num_updates % self.args.log_interval == 0:
                            # to log
                            train_loss_mean = self.log_step(log_dict, tensorboard_suffix='train', epoch=epoch, num_updates=num_updates,
                                        lr=self.scheduler.get_last_lr()[0])
                        self.reset_logging(log_dict)
                        if self.args.valid_interval is not None and \
                                num_updates % self.args.valid_interval == 0:
                            current_metrics = self.valid(epoch, num_updates)
                            best_update, average_score = self.early_stop(current_metrics, best_metric, epoch, num_updates)
                            if not best_update or train_loss_mean < 1e-7:
                                early_stop += 1
                                self.logger.info(f"Early stop + 1 = {early_stop}. " \
                                            f"Best averate score = {best_metric} at {best_num_updates}.")
                            else:
                                early_stop = 0
                                best_metric = average_score
                                best_num_updates = num_updates
                            if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                                break
                        if self.args.output_interval is not None and \
                                num_updates % self.args.output_interval == 0:
                            save_path = f"{self.args.output_dir}/checkpoint@{epoch}-{num_updates}.pt"
                            self.save_checkpoint(save_path, epoch, num_updates)
                            
                        if num_updates >= self.args.train_iters:
                            break
                
                if self.args.early_stop > 0 and early_stop >= self.args.early_stop:
                    self.logger.info(f"Stop training. Best averate score = {best_metric:.3f} at {best_num_updates}.")
                    break
                if num_updates >= self.args.train_iters:
                    break

            if save_last:
                save_path = f"{self.args.output_dir}/checkpoint-last.pt"
                self.save_checkpoint(save_path, epoch, num_updates)


            ###################################  test  ##########################################
            if self.args.test_checkpoint:
                self.load_checkpoint(self.args.test_checkpoint)
                self.model.eval()
                self.logger.info('load checkpoints from'+self.args.test_checkpoint)
            else:
                load_best_path = f"{self.args.output_dir}/checkpoint-best.pt"
                self.load_checkpoint(load_best_path)
                self.model.eval()
                self.logger.info('load checkpoints from'+load_best_path)
            
            test_dataloader = self.test_data.dataloader
            my_index = []
            my_prediction= []
            test_log_dict = OrderedDict()
            self.logger.info("Begin test on {:d} samples ...".format(len(self.test_data.dataset)))
            metrics = {}
            # if self.rank == 0:
                # print(self.model.encoder.block[0].layer[0].SelfAttention.q.weight)
                # print(self.model.training)
                # print(self.model.module.encoder.block[0].layer[0].SelfAttention.q.weight)
            with torch.no_grad():
                for local_step, batch in enumerate(test_dataloader):
                    # if local_step == 0:
                    #     temp = torch.masked_fill(batch['labels'][1], batch['labels'][1] == -100, 0)
                    #     print(self.tokenizer.convert_ids_to_tokens(temp))
                    #     print(self.tokenizer.decode(temp))
                    #     print(self.tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
                    # quit()
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    test_loss = output[0]
                    self.add_logging(test_log_dict, 'loss', test_loss.item())
                    
                    decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                    generated_ids = self.model.generate(
                        inputs_embeds=all_input["inputs_embeds"],
                        attention_mask=all_input["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        # decoder_start_token_id=self.config.decoder_start_token_id,
                        max_length=self.args.max_output_length,
                        early_stopping=True
                    )
                    gen_text = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    gen_text = list(map(str.strip, gen_text))
                    # my_index.extend(batch['id'])
                    my_prediction.extend(gen_text)
            if len(my_prediction) != 0:
                # metrics = self.evaluate(my_prediction, my_index, self.test_dataset)
                metrics = self.test_data.evaluate(my_prediction, verbose=False)
            test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                        epoch=epoch, num_updates=num_updates, **metrics)
            if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
                self.model.train()
            # metrics['loss'] = - test_loss

            for i,j in metrics.items():
                metric = i
                test_performance = j
            with open(out_path+'/result.tsv', 'a') as fout:
                fout.write(task+','+str(best_metric)+','+str(test_performance)+'\n')

        return best_metric, 
    
    def early_stop(self, metrics, best_metric, epoch, num_updates):
        current_metric = 0
        update = True
        for key in metrics:
            current_metric += metrics[key]
        current_metric = current_metric / len(metrics)  # compare average
        if best_metric > current_metric:
            update = False
        else:
            save_path = f"{self.args.output_dir}/checkpoint-best.pt"
            self.save_checkpoint(save_path, epoch, num_updates)
            
        return update, current_metric

    def valid(self, epoch=0, num_updates=0):
        self.model.eval()
        # valid_dataloader, valid_sampler = self.build_dataloader(self.args, self.valid_dataset,
        #                    batch_size=self.args.eval_micro_batch_size, shuffle=False)
        valid_dataloader = self.dev_data.dataloader
        # valid_dataloader = self.test_data.dataloader
        my_index = []
        my_prediction= []
        valid_log_dict = OrderedDict()
        self.logger.info("Begin validation on {:d} samples ...".format(len(self.dev_data.dataset)))
        metrics = {}
        # if self.rank == 0:
            # print(self.model.encoder.block[0].layer[0].SelfAttention.q.weight)
            # print(self.model.training)
            # print(self.model.module.encoder.block[0].layer[0].SelfAttention.q.weight)
        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):
                # if local_step == 0:
                #     temp = torch.masked_fill(batch['labels'][1], batch['labels'][1] == -100, 0)
                #     print(self.tokenizer.convert_ids_to_tokens(temp))
                #     print(self.tokenizer.decode(temp))
                #     print(self.tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
                # quit()
                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                valid_loss = output[0]
                self.add_logging(valid_log_dict, 'loss', valid_loss.item())
                
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                generated_ids = self.model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    # decoder_start_token_id=self.config.decoder_start_token_id,
                    max_length=self.args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                # my_index.extend(batch['id'])
                my_prediction.extend(gen_text)
        if len(my_prediction) != 0:
            # metrics = self.evaluate(my_prediction, my_index, self.valid_dataset)
            metrics = self.dev_data.evaluate(my_prediction, verbose=False)
        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                      epoch=epoch, num_updates=num_updates, **metrics)
        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
            self.model.train()
        # metrics['loss'] = - valid_loss
        return metrics

    def test(self, epoch=0, num_updates=0):
        # if self.args.tune_method == 'lora_stage2':
        #     self.load_lora_B(self.args.load_lora_B_path)
        if self.args.test_checkpoint:
            self.load_checkpoint(self.args.test_checkpoint)
            self.model.eval()
            self.logger.info('load checkpoints from'+self.args.test_checkpoint)
        else:
            load_best_path = f"{self.args.output_dir}/checkpoint-best.pt"
            self.load_checkpoint(load_best_path)
            self.model.eval()
            self.logger.info('load checkpoints from'+load_best_path)
        
        test_dataloader = self.test_data.dataloader
        my_index = []
        my_prediction= []
        test_log_dict = OrderedDict()
        self.logger.info("Begin test on {:d} samples ...".format(len(self.test_data.dataset)))
        metrics = {}
        # if self.rank == 0:
            # print(self.model.encoder.block[0].layer[0].SelfAttention.q.weight)
            # print(self.model.training)
            # print(self.model.module.encoder.block[0].layer[0].SelfAttention.q.weight)
        with torch.no_grad():
            for local_step, batch in enumerate(test_dataloader):
                # if local_step == 0:
                #     temp = torch.masked_fill(batch['labels'][1], batch['labels'][1] == -100, 0)
                #     print(self.tokenizer.convert_ids_to_tokens(temp))
                #     print(self.tokenizer.decode(temp))
                #     print(self.tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
                # quit()
                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                test_loss = output[0]
                self.add_logging(test_log_dict, 'loss', test_loss.item())
                
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                generated_ids = self.model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    # decoder_start_token_id=self.config.decoder_start_token_id,
                    max_length=self.args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                # my_index.extend(batch['id'])
                my_prediction.extend(gen_text)
        if len(my_prediction) != 0:
            # metrics = self.evaluate(my_prediction, my_index, self.test_dataset)
            metrics = self.test_data.evaluate(my_prediction, verbose=False)
        test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                      epoch=epoch, num_updates=num_updates, **metrics)
        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
            self.model.train()
        # metrics['loss'] = - test_loss
        return metrics
    
    def test_all_ckpt(self, epoch=0, num_updates=0):
        from task_list import QA_task_list

        for task in QA_task_list:
            self.args.test_checkpoint = '/data/private/lvxingtai/delta_search_result/prefix_full_data_ckpt_from_yijing/'+task+'/checkpoint-best.pt'
            # self.args.test_checkpoint = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
            #      +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
            # self.args.test_checkpoint = '/data/private/yijing/CrossFit_ensemble/models/full_data_lora/'+task\
            #     +'-lora_size_10-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
            if self.args.test_checkpoint:
                self.load_checkpoint(self.args.test_checkpoint)
                self.model.eval()
                self.logger.info('load checkpoints from'+self.args.test_checkpoint)
            else:
                load_best_path = f"{self.args.output_dir}/checkpoint-best.pt"
                self.load_checkpoint(load_best_path)
                self.model.eval()
                self.logger.info('load checkpoints from'+load_best_path)

            test_dataloader = self.dev_data.dataloader
            my_index = []
            my_prediction= []
            test_log_dict = OrderedDict()
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))
            metrics = {}
            # if self.rank == 0:
                # print(self.model.encoder.block[0].layer[0].SelfAttention.q.weight)
                # print(self.model.training)
                # print(self.model.module.encoder.block[0].layer[0].SelfAttention.q.weight)
            with torch.no_grad():
                for local_step, batch in enumerate(test_dataloader):
                    # if local_step == 0:
                    #     temp = torch.masked_fill(batch['labels'][1], batch['labels'][1] == -100, 0)
                    #     print(self.tokenizer.convert_ids_to_tokens(temp))
                    #     print(self.tokenizer.decode(temp))
                    #     print(self.tokenizer.convert_ids_to_tokens(batch['input_ids'][1]))
                    # quit()
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    test_loss = output[0]
                    self.add_logging(test_log_dict, 'loss', test_loss.item())
                    
                    decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                    generated_ids = self.model.generate(
                        inputs_embeds=all_input["inputs_embeds"],
                        attention_mask=all_input["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        # decoder_start_token_id=self.config.decoder_start_token_id,
                        max_length=self.args.max_output_length,
                        early_stopping=True
                    )
                    gen_text = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    gen_text = list(map(str.strip, gen_text))
                    # my_index.extend(batch['id'])
                    my_prediction.extend(gen_text)
            if len(my_prediction) != 0:
                # metrics = self.evaluate(my_prediction, my_index, self.test_dataset)
                metrics = self.dev_data.evaluate(my_prediction, verbose=False)
            test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                        epoch=epoch, num_updates=num_updates, **metrics)
            if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
                self.model.train()
            # metrics['loss'] = - test_loss
            for i,j in metrics.items():
                metric = i
                test_performance = j
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/prefix/qa_tasks/zeroshot_dev/result.tsv', 'a') as fout:
                fout.write(task+','+str(test_performance)+'\n')
        return metrics

    def get_Loss(self):
        from task_list import QA_task_list as TASK_LIST

        for task in TASK_LIST:
            source_checkpoint = self.args.source_checkpoint_path+'/'+task+'/checkpoint-best.pt'
            
            self.load_checkpoint(source_checkpoint)
            self.model.eval()
            self.logger.info('load checkpoints from'+source_checkpoint)

            dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    test_loss = output[0]
                    break

            with open(self.args.output_dir+'/result.tsv', 'a') as fout:
                fout.write(str(float(test_loss))+'\n')

        return None

    def get_KL_divergence(self, target_task, data_mode):
        #KL_divergence越大，两个分布越不相似,目前的计算方法是，针对每个词的分布（logits）计算KL_divergence，然后将所有
        #“有效词”的KL_divergence求和得到最终的KL_divergence
        import scipy.stats
        import numpy
        import math

        if data_mode == 'test':
            #load data
            test_dataloader = self.test_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.test_data.dataset)))

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            cou = 0
            target_logits_list = []
            target_labels_list = []
            with torch.no_grad():
                for local_step, batch in enumerate(test_dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits_list.append(output[1])
                    target_labels_list.append(batch[2])
                    cou += 1
                    if cou == 2:
                        break
            target_logits = torch.cat((target_logits_list[0], target_logits_list[1]), 0)
            target_labels = torch.cat((target_labels_list[0], target_labels_list[1]), 0)

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                cou = 0
                source_logits_list = []
                source_labels_list = []
                with torch.no_grad():
                    for local_step, batch in enumerate(test_dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits_list.append(output[1])
                        source_labels_list.append(batch[2])
                        cou += 1
                        if cou == 2:
                            break
                source_logits = torch.cat((source_logits_list[0], source_logits_list[1]), 0)
                source_labels = torch.cat((source_labels_list[0], source_labels_list[1]), 0)

                KL_sum = 0
                for i in range(len(target_labels)):
                    for j in range(len(target_labels[i])):
                        if int(target_labels[i][j]) == -100:
                            break
                        P = torch.softmax(target_logits[i][j],0).cpu().numpy()
                        Q = torch.softmax(source_logits[i][j],0).cpu().numpy()
                        zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                        P = numpy.delete(P, zero_index)
                        Q = numpy.delete(Q, zero_index)
                        KL_sum += scipy.stats.entropy(pk=P, qk=Q)

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/KL_divergence/result.tsv', 'a') as fout:
                    fout.write(task+','+str(KL_sum)+'\n')

        if data_mode == 'dev':
            #load data
            test_dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get target logits(use 64_tune ckpt)
            # target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/all_tasks/64shot_random/'+target_task+'_best.pt'
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_256shot/random/'+target_task+'/checkpoint-best.pt'

            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(test_dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list as TASK_LIST
            for task in TASK_LIST:
                # source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                #     +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(test_dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                KL_sum = 0
                for i in range(len(target_labels)):
                    for j in range(len(target_labels[i])):
                        if int(target_labels[i][j]) == -100:
                            break
                        P = torch.softmax(target_logits[i][j],0).cpu().numpy()
                        Q = torch.softmax(source_logits[i][j],0).cpu().numpy()
                        zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                        P = numpy.delete(P, zero_index)
                        Q = numpy.delete(Q, zero_index)
                        KL_sum += scipy.stats.entropy(pk=P, qk=Q)

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_256shot/KL_divergence/result.tsv', 'a') as fout:
                    fout.write(task+','+str(KL_sum)+'\n')

    def get_KL_encoder(self, target_task, data_mode):
        #KL_divergence越大，两个分布越不相似,目前的计算方法是，针对每个词的分布（logits）计算KL_divergence，然后将所有
        #“有效词”的KL_divergence求和得到最终的KL_divergence
        import scipy.stats
        import numpy
        import math
        batch_size = 50

        if data_mode == 'dev_little_gpu':
            #load data
            dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                cou = 0
                target_encoder_state_list = []
                target_labels_list = []
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_encoder_state_list.append(output['encoder_last_hidden_state'].view(batch_size,-1))
                    target_labels_list.append(batch[2])
                    cou += 1
                    if cou == 5:
                        break
                target_encoder_state = torch.cat((target_encoder_state_list[0], target_encoder_state_list[1], target_encoder_state_list[2], target_encoder_state_list[3], target_encoder_state_list[4]), 0)
                target_labels = torch.cat((target_labels_list[0], target_labels_list[1], target_labels_list[2], target_labels_list[3], target_labels_list[4]), 0)

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    cou = 0
                    source_encoder_state_list = []
                    source_labels_list = []
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_encoder_state_list.append(output['encoder_last_hidden_state'].view(batch_size,-1))
                        source_labels_list.append(batch[2])
                        cou += 1
                        if cou == 5:
                            break
                    source_encoder_state = torch.cat((source_encoder_state_list[0], source_encoder_state_list[1], source_encoder_state_list[2], source_encoder_state_list[3], source_encoder_state_list[4]), 0)
                    source_labels = torch.cat((source_labels_list[0], source_labels_list[1], source_labels_list[2], source_labels_list[3], source_labels_list[4]), 0)


                KL_sum = 0
                for i in range(len(target_encoder_state)):
                    P = torch.softmax(target_encoder_state[i],0).cpu().numpy()
                    Q = torch.softmax(source_encoder_state[i],0).cpu().numpy()
                    zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                    P = numpy.delete(P, zero_index)
                    Q = numpy.delete(Q, zero_index)
                    KL_sum += scipy.stats.entropy(pk=P, qk=Q)

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/KL_encoder/result.tsv', 'a') as fout:
                    fout.write(task+','+str(KL_sum)+'\n')

        if data_mode == 'dev':
            from task_list import lalal




            if target_task in ['boolq','mc_taco',]:
                batch_size = 32
            else:
                batch_size = 50

            #load data
            dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_encoder_state = output['encoder_last_hidden_state'].view(batch_size,-1)
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_encoder_state = output['encoder_last_hidden_state'].view(batch_size,-1)
                        source_labels = batch[2]
                        break

                KL_sum = 0
                for i in range(len(target_encoder_state)):
                    P = torch.softmax(target_encoder_state[i],0).cpu().numpy()
                    Q = torch.softmax(source_encoder_state[i],0).cpu().numpy()
                    zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                    P = numpy.delete(P, zero_index)
                    Q = numpy.delete(Q, zero_index)
                    KL_sum += scipy.stats.entropy(pk=P, qk=Q)

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/KL_encoder/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(KL_sum))+'\n')

    def get_block(self, target_task, mode):
        if mode == 'KL':
            # 代码有问题，这个机制也存在问题
            import scipy.stats
            import numpy

            for i in range(12):
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/encoder_block_'+str(i)+'.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/decoder_block_'+str(i)+'.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
            with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/encoder.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/decoder.tsv', 'a') as fout:
                fout.write(target_task+'\n')

            dataloader = self.dev_data.dataloader

            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                for i in range(12):
                    KL_sum = 0
                    for j in range(len(target_encoder_hidden_states[i])):
                        for k in range(len(target_encoder_hidden_states[i][j])):
                            P = torch.softmax(target_encoder_hidden_states[i][j][k],0).cpu().numpy()
                            Q = torch.softmax(source_encoder_hidden_states[i][j][k],0).cpu().numpy()
                            zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                            P = numpy.delete(P, zero_index)
                            Q = numpy.delete(Q, zero_index)
                            KL_sum += scipy.stats.entropy(pk=P, qk=Q)
                    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/encoder_block_'+str(i)+'.tsv', 'a') as fout:
                        fout.write(task+','+str(KL_sum)+'\n')

                    KL_sum = 0
                    for j in range(len(target_decoder_hidden_states[i])):
                        for k in range(len(target_decoder_hidden_states[i][j])):
                            P = torch.softmax(target_decoder_hidden_states[i][j][k],0).cpu().numpy()
                            Q = torch.softmax(source_decoder_hidden_states[i][j][k],0).cpu().numpy()
                            zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                            P = numpy.delete(P, zero_index)
                            Q = numpy.delete(Q, zero_index)
                            KL_sum += scipy.stats.entropy(pk=P, qk=Q)
                            print(len(P))
                    from IPython import embed
                    embed()
                    import sys
                    sys.exit(0)

                    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/decoder_block_'+str(i)+'.tsv', 'a') as fout:
                        fout.write(task+','+str(KL_sum)+'\n')
                
                KL_sum = 0
                for j in range(len(target_encoder_hidden_states[12])):
                    for k in range(len(target_encoder_hidden_states[12][j])):
                        P = torch.softmax(target_encoder_hidden_states[12][j][k],0).cpu().numpy()
                        Q = torch.softmax(source_encoder_hidden_states[12][j][k],0).cpu().numpy()
                        zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                        P = numpy.delete(P, zero_index)
                        Q = numpy.delete(Q, zero_index)
                        KL_sum += scipy.stats.entropy(pk=P, qk=Q)
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/encoder.tsv', 'a') as fout:
                    fout.write(task+','+str(KL_sum)+'\n')
            
                KL_sum = 0
                for j in range(len(target_encoder_hidden_states[12])):
                    for k in range(len(target_encoder_hidden_states[12][j])):
                        P = torch.softmax(target_encoder_hidden_states[12][j][k],0).cpu().numpy()
                        Q = torch.softmax(source_encoder_hidden_states[12][j][k],0).cpu().numpy()
                        zero_index = numpy.union1d(np.where(P==0)[0],np.where(Q==0)[0])
                        P = numpy.delete(P, zero_index)
                        Q = numpy.delete(Q, zero_index)
                        KL_sum += scipy.stats.entropy(pk=P, qk=Q)
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/KL50_block_tem10/KL_block_initial_data/encoder.tsv', 'a') as fout:
                    fout.write(task+','+str(KL_sum)+'\n')
        
        if mode == 'dif_last':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_last.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                dif_sum = torch.norm(target_logits-source_logits)
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_last.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'dif_last_label':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_last_label.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                dif_sum = 0
                for i in range(len(target_labels)):
                    for j in range(len(target_labels[i])):
                        if int(target_labels[i][j]) == -100:
                            break
                        dif_sum += torch.norm(target_logits[i][j]-source_logits[i][j])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_last_label.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'dif_last_label_all':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_last_label_all.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                dif_sum = 0
                for i in range(len(target_labels)):
                    if any(target_labels[i]==-100):
                        j = int(torch.nonzero(target_labels[i]==-100).squeeze(1)[0])
                        dif_sum += torch.norm(target_logits[i,0:j,:]-source_logits[i,0:j,:])
                    else:
                        dif_sum += torch.norm(target_logits[i,:,:]-source_logits[i,:,:])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_last_label_all.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'cos_last':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_last.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                sim = cos_sim(target_logits.view(-1),source_logits.view(-1))

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_last.tsv', 'a') as fout:
                    fout.write(task+','+str(float(sim))+'\n')

        if mode == 'cos_last_label':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_last_label.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_labels)):
                    for j in range(len(target_labels[i])):
                        if int(target_labels[i][j]) == -100:
                            break
                        cos_sum += cos_sim(target_logits[i][j],source_logits[i][j])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_last_label.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'cos_last_label_all':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_last_label_all.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    target_logits = output[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_labels)):
                    if any(target_labels[i]==-100):
                        j = int(torch.nonzero(target_labels[i]==-100).squeeze(1)[0])
                        t_tensor = target_logits[i,0:j,:]
                        s_tensor = source_logits[i,0:j,:]
                        cos_sum += cos_sim(t_tensor.view(-1),s_tensor.view(-1))
                    else:
                        t_tensor = target_logits[i,:,:]
                        s_tensor = source_logits[i,:,:]
                        cos_sum += cos_sim(t_tensor.view(-1),s_tensor.view(-1))

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_last_label_all.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')
        
        if mode == 'dif':
            import scipy.stats
            import numpy

            # for i in range(12):
            #     with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/encoder_block_'+str(i)+'.tsv', 'a') as fout:
            #         fout.write(target_task+'\n')
            #     with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/decoder_block_'+str(i)+'.tsv', 'a') as fout:
            #         fout.write(target_task+'\n')
            # with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/encoder.tsv', 'a') as fout:
            #     fout.write(target_task+'\n')
            # with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/decoder.tsv', 'a') as fout:
            #     fout.write(target_task+'\n')

            dataloader = self.dev_data.dataloader

            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                for i in range(12):
                    dif_sum = torch.norm(target_encoder_hidden_states[i]-source_encoder_hidden_states[i])
                    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/encoder_block_'+str(i)+'.tsv', 'a') as fout:
                        fout.write(task+','+str(float(dif_sum))+'\n')

                    dif_sum = torch.norm(target_decoder_hidden_states[i]-source_decoder_hidden_states[i])
                    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/decoder_block_'+str(i)+'.tsv', 'a') as fout:
                        fout.write(task+','+str(float(dif_sum))+'\n')
                
                dif_sum = torch.norm(target_encoder_hidden_states[12]-source_encoder_hidden_states[12])
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/encoder.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')
            
                dif_sum = torch.norm(target_decoder_hidden_states[12]-source_decoder_hidden_states[12])
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_dif_tem10/dif_initial_data/encoder.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'cos':
            import scipy.stats
            import numpy

            for i in range(12):
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/encoder_block_'+str(i)+'.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/decoder_block_'+str(i)+'.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
            with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/encoder.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/decoder.tsv', 'a') as fout:
                fout.write(target_task+'\n')

            dataloader = self.dev_data.dataloader
            cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                for i in range(12):
                    cos_sum = cos_sim(target_encoder_hidden_states[i].view(-1),source_encoder_hidden_states[i].view(-1))
                    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/encoder_block_'+str(i)+'.tsv', 'a') as fout:
                        fout.write(task+','+str(float(cos_sum))+'\n')

                    cos_sum = cos_sim(target_decoder_hidden_states[i].view(-1),source_decoder_hidden_states[i].view(-1))
                    with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/decoder_block_'+str(i)+'.tsv', 'a') as fout:
                        fout.write(task+','+str(float(cos_sum))+'\n')
                
                cos_sum = cos_sim(target_encoder_hidden_states[12].view(-1),source_encoder_hidden_states[12].view(-1))
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/encoder.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')
            
                cos_sum = cos_sim(target_decoder_hidden_states[12].view(-1),source_decoder_hidden_states[12].view(-1))
                with open('/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/block_cos_tem/cos_initial_data/encoder.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'decoder_dif_last':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_labellast.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_decoder_hidden_states[12])):
                    dif_sum += torch.norm(target_decoder_hidden_states[12][i][63]-source_decoder_hidden_states[12][i][63])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_labellast.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'decoder_cos_last':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_labellast.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_decoder_hidden_states[12])):
                    cos_sum += cos_sim(target_decoder_hidden_states[12][i][63],source_decoder_hidden_states[12][i][63])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_labellast.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')     

        if mode == 'decoder_dif_label':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_label.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_labels)):
                    if any(target_labels[i]==-100):
                        j = int(torch.nonzero(target_labels[i]==-100).squeeze(1)[0])
                        dif_sum += torch.norm(target_decoder_hidden_states[12][i,0:j,:]-source_decoder_hidden_states[12][i,0:j,:])
                    else:
                        dif_sum += torch.norm(target_decoder_hidden_states[12][i,:,:]-source_decoder_hidden_states[12][i,:,:])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_label.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'decoder_cos_label':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_label.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_labels)):
                    if any(target_labels[i]==-100):
                        j = int(torch.nonzero(target_labels[i]==-100).squeeze(1)[0])
                        t_tensor = target_decoder_hidden_states[12][i,0:j,:]
                        s_tensor = source_decoder_hidden_states[12][i,0:j,:]
                        cos_sum += cos_sim(t_tensor.view(-1),s_tensor.view(-1))
                    else:
                        t_tensor = target_decoder_hidden_states[12][i,:,:]
                        s_tensor = source_decoder_hidden_states[12][i,:,:]
                        cos_sum += cos_sim(t_tensor.view(-1),s_tensor.view(-1))

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_label.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n') 

        if mode == 'encoder_dif_withoutpad':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_encoder_withoutpad.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    target_attention_masks = batch[1]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_attention_masks)):
                    if any(target_attention_masks[i]==0):
                        j = int(torch.nonzero(target_attention_masks[i]==0).squeeze(1)[0])
                        dif_sum += torch.norm(target_encoder_hidden_states[12][i,0:j,:]-source_encoder_hidden_states[12][i,0:j,:])
                    else:
                        dif_sum += torch.norm(target_encoder_hidden_states[12][i,:,:]-source_encoder_hidden_states[12][i,:,:])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_encoder_withoutpad.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'encoder_cos_withoutpad':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_encoder_withoutpad.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    target_attention_masks = batch[1]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_attention_masks)):
                    if any(target_attention_masks[i]==0):
                        j = int(torch.nonzero(target_attention_masks[i]==0).squeeze(1)[0])
                        t_tensor = target_encoder_hidden_states[12][i,0:j,:]
                        s_tensor = source_encoder_hidden_states[12][i,0:j,:]
                        cos_sum += cos_sim(t_tensor.view(-1),s_tensor.view(-1))
                    else:
                        t_tensor = target_encoder_hidden_states[12][i,:,:]
                        s_tensor = source_encoder_hidden_states[12][i,:,:]
                        cos_sum += cos_sim(t_tensor.view(-1),s_tensor.view(-1))

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_encoder_withoutpad.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'encoder_dif_first':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_encoder_first.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_encoder_hidden_states[12])):
                    dif_sum += torch.norm(target_encoder_hidden_states[12][i][0]-source_encoder_hidden_states[12][i][0])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_encoder_first.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'encoder_cos_first':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_encoder_first.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_encoder_hidden_states[12])):
                    cos_sum += cos_sim(target_encoder_hidden_states[12][i][0],source_encoder_hidden_states[12][i][0])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_encoder_first.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'decoder_dif_first':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_first.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_decoder_hidden_states[12])):
                    dif_sum += torch.norm(target_decoder_hidden_states[12][i][0]-source_decoder_hidden_states[12][i][0])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_first.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'decoder_cos_first':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_first.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_decoder_hidden_states[12])):
                    cos_sum += cos_sim(target_decoder_hidden_states[12][i][0],source_decoder_hidden_states[12][i][0])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_first.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'encoder_dif_all':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_encoder_all.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_encoder_hidden_states[12])):
                    dif_sum += torch.norm(target_encoder_hidden_states[12][i]-source_encoder_hidden_states[12][i])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_encoder_all.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'encoder_cos_all':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_encoder_all.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_encoder_hidden_states[12])):
                    cos_sum += cos_sim(target_encoder_hidden_states[12][i].view(-1),source_encoder_hidden_states[12][i].view(-1))

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_encoder_all.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'decoder_dif_all':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_all.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                dif_sum = 0
                for i in range(len(target_decoder_hidden_states[12])):
                    dif_sum += torch.norm(target_decoder_hidden_states[12][i]-source_decoder_hidden_states[12][i])

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_decoder_all.tsv', 'a') as fout:
                    fout.write(task+','+str(float(dif_sum))+'\n')

        if mode == 'decoder_cos_all':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_all.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            dataloader = self.dev_data.dataloader

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/data/private/lvxingtai/checkpoints_from_a100/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sum = 0
                for i in range(len(target_decoder_hidden_states[12])):
                    cos_sum += cos_sim(target_decoder_hidden_states[12][i].view(-1),source_decoder_hidden_states[12][i].view(-1))

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/cos_decoder_all.tsv', 'a') as fout:
                    fout.write(task+','+str(float(cos_sum))+'\n')

        if mode == 'dif_withoutpad':
            for i in range(12):
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_withoutpad/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_withoutpad/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')

            dataloader = self.dev_data.dataloader
            #get target logits(use 64_tune ckpt)
            target_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks/64_tune/'+target_task+'_best.pt'
            self.load_checkpoint(target_ckpt_path)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input,output_hidden_states=True)
                    target_encoder_hidden_states = output['encoder_hidden_states']
                    target_decoder_hidden_states = output['decoder_hidden_states']
                    target_attention_masks = batch[1]
                    target_labels = batch[2]
                    break

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                dif_sum_list = [0 for i in range(24)]
                for i in range(len(target_attention_masks)):
                    if any(target_attention_masks[i]==0):
                        j = int(torch.nonzero(target_attention_masks[i]==0).squeeze(1)[0])
                        for block_num in range(12):
                            dif_sum_list[block_num] += torch.norm(target_encoder_hidden_states[block_num][i,0:j,:]-source_encoder_hidden_states[block_num][i,0:j,:])
                    else:
                        for block_num in range(12):
                            dif_sum_list[block_num] += torch.norm(target_encoder_hidden_states[block_num][i,:,:]-source_encoder_hidden_states[block_num][i,:,:])

                for i in range(len(target_labels)):
                    if any(target_labels[i]==-100):
                        j = int(torch.nonzero(target_labels[i]==-100).squeeze(1)[0])
                        for block_num in range(12):
                            dif_sum_list[block_num+12] += torch.norm(target_decoder_hidden_states[block_num][i,0:j,:]-source_decoder_hidden_states[block_num][i,0:j,:])
                    else:
                        for block_num in range(12):
                            dif_sum_list[block_num+12] += torch.norm(target_decoder_hidden_states[block_num][i,:,:]-source_decoder_hidden_states[block_num][i,:,:])

                for i in range(12):
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_withoutpad/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(dif_sum_list[i]))+'\n')
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/block/dif_withoutpad/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(dif_sum_list[i+12]))+'\n')


    def get_EL2N(self, target_task, data_mode):
        #EL2N的计算方法：对于每句话，根据输入得到某个token的概率输出p，y是这个token对应的one_hot向量，计算||p-y||2

        if data_mode == 'test':
            #load data
            test_dataloader = self.test_data.dataloader

            #get source logits and cum KL_divergence
            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.logger.info('load checkpoints from'+source_ckpt_path)
                self.model.eval()

                cou = 0
                source_logits_list = []
                source_labels_list = []
                with torch.no_grad():
                    for local_step, batch in enumerate(test_dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits_list.append(output[1])
                        source_labels_list.append(batch[2])
                        cou += 1
                        if cou == 2:
                            break
                source_logits = torch.cat((source_logits_list[0], source_logits_list[1]), 0)
                source_labels = torch.cat((source_labels_list[0], source_labels_list[1]), 0)

                EL2N_sum = 0
                for i in range(len(source_labels)):
                    for j in range(len(source_labels[i])):
                        if int(source_labels[i][j]) == -100:
                            break
                        P = torch.softmax(source_logits[i][j],0)
                        P[source_labels[i][j]] -= 1
                        EL2N_sum += torch.norm(P)

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/EL2N/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(EL2N_sum))+'\n')

        if data_mode == 'dev':
            #load data
            test_dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get source logits and cum KL_divergence
            from task_list import QA_task_list as TASK_LIST
            for task in TASK_LIST:
                # source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                #     +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.logger.info('load checkpoints from'+source_ckpt_path)
                self.model.eval()

                source_logits_list = []
                source_labels_list = []
                with torch.no_grad():
                    for local_step, batch in enumerate(test_dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits_list.append(output[1])
                        source_labels_list.append(batch[2])
                        break
                source_logits = source_logits_list[0]
                source_labels = source_labels_list[0]

                EL2N_sum = 0
                for i in range(len(source_labels)):
                    for j in range(len(source_labels[i])):
                        if int(source_labels[i][j]) == -100:
                            break
                        P = torch.softmax(source_logits[i][j],0)
                        P[source_labels[i][j]] -= 1
                        EL2N_sum += torch.norm(P)

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_256shot/EL2N/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(EL2N_sum))+'\n')

    def get_logits_label_cos(self, target_task, data_mode):
        if data_mode == 'dev':

            #load data
            test_dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get source logits and cum KL_divergence
            from task_list import QA_task_list as TASK_LIST
            for task in TASK_LIST:
                # source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                #     +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.logger.info('load checkpoints from'+source_ckpt_path)
                self.model.eval()

                source_logits_list = []
                source_labels_list = []
                with torch.no_grad():
                    for local_step, batch in enumerate(test_dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits_list.append(output[1])
                        source_labels_list.append(batch[2])
                        torch.cuda.empty_cache()

                source_logits = []
                source_labels = []
                for i in range(len(source_logits_list)):
                    source_logits.extend(source_logits_list[i])
                    source_labels.extend(source_labels_list[i])

                sim_sum = 0
                for i in range(len(source_labels)):
                    for j in range(len(source_labels[i])):
                        if int(source_labels[i][j]) == -100:
                            break
                        P = torch.softmax(source_logits[i][j],0)
                        onehot_label = torch.zeros(P.size()).to(self.device)
                        onehot_label[source_labels[i][j]] += 1
                        cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                        sim = cos_sim(P,onehot_label)
                        sim_sum += sim

                import gc
                gc.collect()
                torch.cuda.empty_cache()

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_2048shot/logits_label_cos/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(sim_sum))+'\n')
        
        if data_mode == 'dev_batch':

            #load data
            test_dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get source logits and cum KL_divergence
            from task_list import QA_task_list as TASK_LIST
            for task in TASK_LIST:
                # source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                #     +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_128shot/fulldata/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)
                self.logger.info('load checkpoints from'+source_ckpt_path)
                self.model.eval()

                sim_sum = 0
                with torch.no_grad():
                    for local_step, batch in enumerate(test_dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input)
                        source_logits = output[1]
                        source_labels = batch[2]

                        for i in range(len(source_labels)):
                            for j in range(len(source_labels[i])):
                                if int(source_labels[i][j]) == -100:
                                    break
                                P = torch.softmax(source_logits[i][j],0)
                                onehot_label = torch.zeros(P.size()).to(self.device)
                                onehot_label[source_labels[i][j]] += 1
                                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                                sim = cos_sim(P,onehot_label)
                                sim_sum += sim

                    # import gc
                    # gc.collect()
                    # torch.cuda.empty_cache()

                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_2048shot/logits_label_cos/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(sim_sum))+'\n')
        
    
    def get_GraNd(self, target_task, mode):
        from math import sqrt

        if mode == 'layer':
            #单独计算每个layer的结果，没有叠加
            for i in range(12):
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/encoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/encoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_layer_2_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
            if target_task in ['boolq','mc_taco',]:
                cou_num = 3
            else:
                cou_num = 5 

            dataloader = self.dev_data.dataloader

            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/lvxingtai/delta_search_result/adapter_full_data_ckpt_from_yijing/'+task \
                    +'/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)

                self.model.train()
                optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
                self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
                
                cou = 0
                GraNd_sum_list = [0 for i in range(60)]

                for local_step, batch in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    loss = self.train_step(batch)

                    #cum GraNd
                    one_GraNd_square_list = [0 for i in range(60)]
                    layer_num = 0
                    for n, p in self.model.named_parameters():
                        if "adapter" in n:
                            Grad = float(torch.norm(p.grad))
                            index = layer_num//2
                            one_GraNd_square_list[index] += Grad*Grad
                            layer_num +=1

                    for j in range(len(GraNd_sum_list)):
                        GraNd_sum_list[j] += sqrt(one_GraNd_square_list[j])
                    cou += 1
                    if cou == cou_num:
                        break

                for i in range(12):
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/encoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[2*i]))+'\n')
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/encoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[2*i+1]))+'\n')
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[24+3*i]))+'\n')
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[25+3*i]))+'\n')
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_layer_2_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[26+3*i]))+'\n')
        
        elif mode == 'block':
            #单独计算每个block的结果，没有叠加
            for i in range(12):
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
            if target_task in ['boolq','mc_taco',]:
                cou_num = 3
            else:
                cou_num = 5 

            dataloader = self.dev_data.dataloader

            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)

                self.model.train()
                optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
                self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
                
                cou = 0
                GraNd_sum_list = [0 for i in range(24)]

                for local_step, batch in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    loss = self.train_step(batch)

                    #cum GraNd
                    one_GraNd_square_list = [0 for i in range(24)]
                    for n, p in self.model.named_parameters():
                        if "adapter" in n:
                            Grad = float(torch.norm(p.grad))
                            index = 0
                            if 'decoder' in n:
                                index += 12
                            index += int(float(n[14:16]))
                            one_GraNd_square_list[index] += Grad*Grad

                    for j in range(len(GraNd_sum_list)):
                        GraNd_sum_list[j] += sqrt(one_GraNd_square_list[j])
                    cou += 1
                    if cou == cou_num:
                        break

                for i in range(12):
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[i]))+'\n')
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[i+12]))+'\n')
        
        elif mode == 'without_eli':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/result.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            if target_task in ['boolq','mc_taco',]:
                cou_num = 6
            else:
                cou_num = 10 

            #load data
            dataloader = self.dev_data.dataloader

            from task_list import just_task as TASK_LIST
            for task in TASK_LIST:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)

                self.model.train()
                optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
                self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
                
                cou = 0
                GraNd_sum = 0
                for local_step, batch in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    loss = self.train_step(batch)
                    #cum GraNd
                    one_GraNd_square = 0
                    for n, p in self.model.named_parameters():
                        if "adapter" in n:
                            Grad = float(torch.norm(p.grad))
                            one_GraNd_square += Grad*Grad
                    GraNd_sum += sqrt(one_GraNd_square)
                    cou += 1
                    if cou == cou_num:
                        break
                
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(GraNd_sum))+'\n')

        elif mode == 'dev':
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_large/GraNd/result.tsv', 'a') as fout:
                fout.write(target_task+'\n')
            if target_task in ['boolq','mc_taco','amazon_polarity','tab_fact','scitail','tweet_eval-offensive','tweet_eval-irony','glue-mrpc','glue-qqp','medical_questions_pairs','blimp-ellipsis_n_bar_1','blimp-irregular_past_participle_adjectives','blimp-sentential_negation_npi_scope',]:
                cou_num = 6
            elif target_task in ['financial_phrasebank','anli','tweet_eval-sentiment',]:
                cou_num = 9
            elif target_task in ['climate_fever','health_fact',]:
                cou_num = 12
            elif target_task in ['liar',]:
                cou_num = 18
            else:
                cou_num = 10 

            #load data
            dataloader = self.dev_data.dataloader

            from task_list import QA_task_list as TASK_LIST
            for task in TASK_LIST:
                # source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                #     +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                source_ckpt_path = '/home/lvxingtai/lxt/crossfit_yijing/checkpoints/adapter/qa_tasks_large/fulldata/'+task \
                    +'-adapter_size_12-seed_44_t5_large/lr_0.0005_bsz_16_seed_44/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)

                self.model.train()
                optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
                self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
                
                cou = 0
                GraNd_sum = 0
                for local_step, batch in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    loss = self.train_step(batch)
                    #cum GraNd
                    one_GraNd_square = 0
                    for n, p in self.model.named_parameters():
                        if "adapter" in n:
                            Grad = float(torch.norm(p.grad))
                            one_GraNd_square += Grad*Grad
                    GraNd_sum += sqrt(one_GraNd_square)
                    cou += 1
                    if cou == cou_num:
                        break
                
                with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks_large/GraNd/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(GraNd_sum))+'\n')
        
        else:
            with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/result.tsv', 'a') as fout:
                fout.write(target_task+'\n')

            #load data
            dataloader = self.test_data.dataloader

            from task_list import QA_task_list
            for task in QA_task_list:
                source_ckpt_path = '/data/private/yijing/CrossFit_ensemble/models/full_data_adapter/'+task \
                    +'-adapter_size_12-seed_42/lr_0.0005_bsz_16_seed_42/checkpoint-best.pt'
                self.load_checkpoint(source_ckpt_path)

                self.model.train()
                optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
                self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)

                if mode == 'old':
                    #这个计算模式应该有点问题，实验也证明不如new要好，以后的mode都是默认使用new
                    cou = 0
                    GraNd_sum = 0
                    for local_step, batch in enumerate(dataloader):
                        self.optimizer.zero_grad()
                        loss = self.train_step(batch)
                        #cum GraNd
                        for n, p in self.model.named_parameters():
                            if "adapter" in n:
                                GraNd_sum += torch.norm(p.grad)
                        cou += 1
                        if cou == 25:
                            break
                    
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum))+'\n')
                
                if mode == 'new':
                    cou = 0
                    GraNd_sum = 0
                    for local_step, batch in enumerate(dataloader):
                        self.optimizer.zero_grad()
                        loss = self.train_step(batch)
                        #cum GraNd
                        one_GraNd_square = 0
                        for n, p in self.model.named_parameters():
                            if "adapter" in n:
                                print(n)
                                Grad = float(torch.norm(p.grad))
                                one_GraNd_square += Grad*Grad
                        GraNd_sum += sqrt(one_GraNd_square)
                        cou += 1
                        if cou == 40:
                            break
                    
                    with open('/home/lvxingtai/lxt/crossfit_yijing/result/adapter/qa_tasks/GraNd/result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum))+'\n')


    def get_decoder_input_ids(self, inputs_embeds):
        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_input_ids = (
                torch.ones((inputs_embeds.shape[0], 1), dtype=torch.long, device=inputs_embeds.device) * decoder_start_token_id
        )
        return decoder_input_ids
    
    def save_checkpoint(self, path, epoch, num_updates):
        state_dict = OrderedDict()
        if self.args.tune_method == 'model':
            # don't save model
            state_dict['model'] = self.model.state_dict()
        elif self.args.tune_method == 'fastfood':
            model_state_dict = self.model.state_dict()
            model_state_dict['projection_params'] = self.ID_wrap.projection_params
            state_dict['fastfood'] = model_state_dict
        elif self.args.tune_method == 'prompt':
            # save prompt
            state_dict['prompt'] = self.prompt
        elif self.args.tune_method == 'lora' or self.args.tune_method == 'bias':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['lora'] = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
        elif self.args.tune_method == 'adapter':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['adapter'] = {k: my_state_dict[k] for k in my_state_dict if 'adapter_' in k}
        elif self.args.tune_method == 'prefix':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['prefix'] = {k: my_state_dict[k] for k in my_state_dict if 'prefix' in k}
        elif self.args.tune_method == 'hyper_PET':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['hyper_PET'] = {k: my_state_dict[k] for k in my_state_dict if 'intrinsic' in k or 'hyper' in k}
        elif self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias_stage2':
            # save lora
            my_state_dict = self.model.state_dict()
            state_dict['lora_R'] = {k: my_state_dict[k] for k in my_state_dict if 'lora_R' in k}
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        state_dict['config'] = self.config
        state_dict['args'] = vars(self.args)
        state_dict['current_state'] = {'epoch': epoch, 'num_updates': num_updates}
        torch.save(state_dict, path)
        self.logger.info(f"epoch: {epoch} num_updates: {num_updates} Save {self.args.tune_method} to {path}.")
    
    def load_from_nearest(self, bs_lr_dir):
        best_ckpt_path = bs_lr_dir + '/checkpoint-best.pt'
        if not os.path.exists(best_ckpt_path):
            return 0, 0, -1
            # assert False, f'No checkpoint-best.pt in {bs_lr_dir}'
        else:
            best_state_dict = torch.load(best_ckpt_path)
            best_epoch = best_state_dict['current_state']['epoch']
            best_num_updates = best_state_dict['current_state']['num_updates']
            self.logger.info(f'Validing for {best_ckpt_path}')
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in best_state_dict[best_state_dict['args']['tune_method']].items()})
            self.model.load_state_dict(model_dict)
            current_metrics = self.valid(best_epoch, best_num_updates)
            best_average_score = 0
            for key in current_metrics:
                best_average_score += current_metrics[key]
            best_average_score = best_average_score / len(current_metrics)  # compare average
            self.logger.info('step: ' + str(best_num_updates) + ' best_accuracy: ' + str(best_average_score))

        max_1w_num_updates = -1
        all_file_name = os.listdir(bs_lr_dir)
        all_1w_ckpt_name = []
        for file_name in all_file_name:
            if ".pt" in file_name and 'best' not in file_name:
                all_1w_ckpt_name.append(file_name)
        if len(all_1w_ckpt_name)>0:
            all_save_step_to_ckpt_name = {}
            for each_1w_ckpt_name in all_1w_ckpt_name:
                each_save_step = int(each_1w_ckpt_name.split('-')[-1].split('.')[0])
                print(each_save_step)
                all_save_step_to_ckpt_name[each_save_step] = each_1w_ckpt_name
            # 在all_1w_ckpt_name中找最大的一个
            max_1w_ckpt_name = all_save_step_to_ckpt_name[max(all_save_step_to_ckpt_name.keys())]
            max_1w_ckpt_path = bs_lr_dir + '/' + max_1w_ckpt_name
            max_state_dict = torch.load(max_1w_ckpt_path)
            max_1w_num_updates = max_state_dict['current_state']['num_updates']
            assert max_1w_num_updates==max(all_save_step_to_ckpt_name.keys()), 'False max_1w_num_updates'
        
        
        self.logger.info(f'best step: {best_num_updates};  max step: {max_1w_num_updates}')
        if best_num_updates >= max_1w_num_updates:
            nearest_ckpt_path = best_ckpt_path
        else:
            nearest_ckpt_path = max_1w_ckpt_path
        
        state_dict = torch.load(nearest_ckpt_path)
        self.logger.info(f"Loading from {nearest_ckpt_path}")
        save_tune_method = state_dict['args']['tune_method']
        if save_tune_method == self.args.tune_method:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict[save_tune_method].items()})
            self.model.load_state_dict(model_dict)

            epoch_and_step = state_dict['current_state']
            epoch = epoch_and_step['epoch']
            num_updates = epoch_and_step['num_updates']
            
        else:
            assert False, "False tune_method or path!"
        
        # 之前忘记存valid_best_accuracy了...所以这里测一下，希望和之前的一样
        
        return epoch, num_updates, best_average_score

    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        if state_dict['args']['tune_method'] == 'model':
            # load model
            self.model.load_state_dict(state_dict['model'])
        elif state_dict['args']['tune_method'] == 'fastfood':
            # load model
            input()
            self.model.load_state_dict(state_dict['fastfood'])
        elif state_dict['args']['tune_method'] == 'prompt':
            # load prompt
            self.prompt = state_dict['prompt']
        elif state_dict['args']['tune_method'] == 'lora' or state_dict['args']['tune_method'] == 'bias':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'adapter':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['adapter'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'prefix':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['prefix'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'hyper_PET':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['hyper_PET'].items()})
            self.model.load_state_dict(model_dict)
        elif state_dict['args']['tune_method'] == 'lora_stage2' or state_dict['args']['tune_method'] == 'bias_stage2':
            # load lora
            model_dict = {k: v for (k, v) in self.model.state_dict().items()}
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora_R'].items()})
            self.model.load_state_dict(model_dict)    
        current_state = state_dict['current_state']
        self.logger.info(f"Load {state_dict['args']['tune_method']} from {path}.")
        return current_state

    def load_lora_B(self, path):
        state_dict = torch.load(path)
        model_dict = {k: v for (k, v) in self.model.state_dict().items()}
        if self.args.decoder_mlp:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k or 'lora_C' in k})
        else:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k})
        self.model.load_state_dict(model_dict)
        
    def load_bias(self, path):
        state_dict = torch.load(path)
        model_dict = {k: v for (k, v) in self.model.state_dict().items()}
        if self.args.decoder_mlp:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k or 'lora_C' in k})
        else:
            model_dict.update({k: v.cuda() for (k, v) in state_dict['lora'].items() if 'lora_B' in k})
        self.model.load_state_dict(model_dict)
    
    def build_optimizer(self, args, params):
        # optimizer = Adafactor(params, lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
        # optimizer = AdamW(params, lr=args.learning_rate)
        optimizer = Adafactor(params, lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
        return optimizer


    def prepare_model_data(self, batch): # t5的输入input_ids全部转化为input_embeds
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self.model.get_input_embeddings()(input_ids)
        all_input['inputs_embeds'] = input_embeds
        # batch[0], batch[1] = trim_batch(batch[0], self.tokenizer.pad_token_id, batch[1])
        # all_input['labels'], all_input['decoder_attention_mask'] = trim_batch(all_input['labels'], self.tokenizer.pad_token_id, all_input['decoder_attention_mask'])
        return all_input

    def prepare_prompt_data(self, batch):
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self.model.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        prompt = torch.unsqueeze(self.prompt, dim=0).expand((batch_size,) + self.prompt.shape)
        prompt_attention = torch.ones(prompt.shape[:2], dtype=torch.long, device=prompt.device)
        # cat prompt with input ids
        input_embeds = torch.cat((prompt, input_embeds), dim=1)
        # cat prompt attention mask to initial attention mask
        all_input['attention_mask'] = torch.cat((prompt_attention, all_input['attention_mask']), dim=1)
        # print("input_embeds", input_embeds.shape)
        all_input['inputs_embeds'] = input_embeds
        # all_input['labels'], all_input['decoder_attention_mask'] = trim_batch(all_input['labels'], self.tokenizer.pad_token_id, all_input['decoder_attention_mask'])
        return all_input

    def train_step(self, batch):
        all_input = self.prepare_data(batch)
        output = self.model(**all_input)
        loss = output[0] / self.gradient_accumulation_steps
        # loss.backward(retain_graph=True)
        loss.backward()
        return loss
    
    def optimizer_step(self, parameters):
        updated = True
        scale = 0
        
        torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)
        self.optimizer.step()
        if updated:
            self.scheduler.step()
        #只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整
        self.optimizer.zero_grad()
        return updated, scale
    
    def log_step(self, log_dict, suffix='', tensorboard_suffix=None, **kwargs):
        new_log_dict = OrderedDict()
        for key, value in kwargs.items():
            new_log_dict[key] = value
        for key in log_dict:
            key_tensor = torch.tensor(log_dict[key], device=self.device)
            
            key_value = key_tensor.mean().item()
            new_log_dict[key] = key_value
        message = '' + suffix
        # if 'loss' in new_log_dict and self.args.task == 'language-model':
        #     new_log_dict['ppl'] = 2 ** new_log_dict['loss']
        for key, value in new_log_dict.items():
            if isinstance(value, float):
                message += ' {:s}: {:.10f}'.format(key, value)
            else:
                message += ' {:s}: {}'.format(key, value)
        self.logger.info(message)
        # if self.tensorboard is not None:
        #     for key, value in new_log_dict.items():
        #         if key in ['epoch', 'num_updates']:
        #             continue
        #         tag = f'{tensorboard_suffix}/{key}' if tensorboard_suffix is not None else key
        #         global_step = kwargs.get('num_updates', None)
        #         self.tensorboard.add_scalar(tag, value, global_step=global_step)
        return new_log_dict.get('loss', None)
    
    def add_logging(self, log_dict, key, value):
        if key not in log_dict:
            log_dict[key] = []
        log_dict[key].append(value)
    
    def reset_logging(self, log_dict):
        for key in log_dict:
            log_dict[key] = []
