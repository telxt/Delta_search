from hashlib import algorithms_available
import os
import numpy as np
import torch
import logging
import random
import math
import warnings

from transformers import AutoTokenizer, BartTokenizer, BartConfig
from transformers import (
    AdamW,
    Adafactor,
    get_scheduler,
    get_linear_schedule_with_warmup,
    is_torch_available,
)

from dataloader.fewshot_gym_singletask_t5 import NLPFewshotGymSingleTaskData
from transformers import T5ForConditionalGeneration
from intrinsic import intrinsic_dimension, intrinsic_dimension_said
from utils import freeze_embeds, trim_batch

from tqdm import tqdm
from collections import OrderedDict
import itertools
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

def uniform_init(prompt, a=0.0, b=1.0):
    torch.nn.init.uniform_(prompt, a, b)

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
    return params

class Trainer:
    def __init__(self, args, logger, model_provider):
        self.args = args
        self.logger = logger
        logger.info("Loading model ...")
        
        self.model, self.config, self.tokenizer = model_provider(args)
        if self.args.tune_method == 'fastfood':
            self.model, self.ID_wrap = intrinsic_dimension(self.model, args.intrinsic_dim, None, set(), args.projection_type, "cuda")
        
        logger.info("Loading Dataset ...")
        self.train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
        self.train_data.load_dataset(self.tokenizer)
        self.train_data.load_dataloader()
        self.dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)
        self.dev_data.load_dataset(self.tokenizer)
        self.dev_data.load_dataloader()
        self.test_data = NLPFewshotGymSingleTaskData(logger, args, args.test_file, data_type="test", is_training=False)
        self.test_data.load_dataset(self.tokenizer)
        self.test_data.load_dataloader()

        self.device = self.init_device(args)
        self.model = self.model.to(self.device)
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        
        if args.seed is not None:
            set_seed(args.seed)
        if args.tune_method == 'prompt':
            self.prompt = torch.rand((args.prompt_num, self.config.d_model), requires_grad=True, device=self.device)
            self.prepare_data = self.prepare_prompt_data
            uniform_init(prompt=self.prompt, a=-math.sqrt(1 / self.config.d_model), b=math.sqrt(1 / self.config.d_model))
        elif args.tune_method == 'model' or args.tune_method == 'fastfood':
            self.prepare_data = self.prepare_model_data
        elif args.tune_method == 'lora' or args.tune_method == 'bias' or args.tune_method == 'prefix' or args.tune_method == 'adapter' or args.tune_method == 'hyper_PET':
            self.prepare_data = self.prepare_model_data
        elif args.tune_method == 'lora_stage2' or 'bias_stage2':
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
        args.tensorboard_dir = args.output_dir + '/tensorboard'
        self.tensorboard = SummaryWriter(log_dir=args.tensorboard_dir)

    def get_optimzied_group(self):
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
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
        
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
                    optimizer_grouped_parameters.append({'params': [p]})

            to_update = self.model.parameters()
            sum = 0
            for group in optimizer_grouped_parameters:
                for param in group['params']:
                    sum+=param.numel()
            print(sum)
        elif self.args.tune_method == 'adapter':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "adapter" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "adapter" in n:
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
                p.requires_grad = False
                if "prefix" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "prefix" in n:
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
                p.requires_grad = False
                if "intrinsic" in n or 'hyper' in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "intrinsic" in n or 'hyper' in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        
        elif self.args.tune_method == 'bias_stage2' or  self.args.tune_method =='lora_stage2':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "lora_R" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora_R" in n:
                    optimizer_grouped_parameters.append({'params': [p]})
            to_update = self.model.parameters()
        elif self.args.tune_method == 'bias':
            no_decay = ["bias", "layer_norm.weight"]
            for n, p in self.model.named_parameters():
                p.requires_grad = False
                if "lora" in n:
                    p.requires_grad = True
                    print(n)
            optimizer_grouped_parameters = []
            for n,p in self.model.named_parameters():
                if "lora" in n:
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

        optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()

        self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
        warm_up_steps = self.args.warmup_steps
        self.scheduler = get_scheduler('constant', self.optimizer, warm_up_steps, self.args.train_iters)

        
        num_updates = 0
        log_dict = OrderedDict()
        best_metric = 0
        best_metric_dict = None
        best_num_updates = 0
        early_stop = 0

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

        return best_metric

    
    def train_and_test_on_all_ckpt(self, out_path, save_last):
        from task_list import QA_task_list as TASK_LIST

        for task in TASK_LIST:
            train_checkpoint_path = self.args.train_checkpoint + '/' + task +'/checkpoint-best.pt'

            self.args.output_dir = out_path + '/' + task
            if not os.path.exists(self.args.output_dir):
                os.mkdir(self.args.output_dir)

            if train_checkpoint_path:
                self.load_checkpoint(train_checkpoint_path)
                self.logger.info('load checkpoints from'+train_checkpoint_path)

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

            optimizer_grouped_parameters, to_update_parameters = self.get_optimzied_group()
            self.optimizer = self.build_optimizer(self.args, optimizer_grouped_parameters)
            warm_up_steps = self.args.warmup_steps
            self.scheduler = get_scheduler('constant', self.optimizer, warm_up_steps, self.args.train_iters)
            
            num_updates = 0
            log_dict = OrderedDict()
            best_metric = 0
            best_metric_dict = None
            best_num_updates = 0
            early_stop = 0

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
            with torch.no_grad():
                for local_step, batch in enumerate(test_dataloader):
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    test_loss = output[0]
                    self.add_logging(test_log_dict, 'loss', test_loss.item())
                    
                    decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                    generated_ids = self.model.generate(
                        inputs_embeds=all_input["inputs_embeds"],
                        attention_mask=all_input["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.args.max_output_length,
                        early_stopping=True
                    )
                    gen_text = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    gen_text = list(map(str.strip, gen_text))
                    my_prediction.extend(gen_text)
            if len(my_prediction) != 0:
                metrics = self.test_data.evaluate(my_prediction, verbose=False)
            test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                        epoch=epoch, num_updates=num_updates, **metrics)
            if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
                self.model.train()

            for i,j in metrics.items():
                test_performance = j
            with open(out_path+'/result.tsv', 'a') as fout:
                fout.write(task+','+str(best_metric)+','+str(test_performance)+'\n')

        return best_metric, 
    
    def test_all_ckpt(self, epoch=0, num_updates=0):
        from task_list import QA_task_list

        for task in QA_task_list:
            self.test_checkpoint = self.args.test_checkpoint + '/' + task +'/checkpoint-best.pt'
            if self.test_checkpoint:
                self.load_checkpoint(self.test_checkpoint)
                self.model.eval()
                self.logger.info('load checkpoints from'+self.test_checkpoint)
            else:
                load_best_path = f"{self.args.output_dir}/checkpoint-best.pt"
                self.load_checkpoint(load_best_path)
                self.model.eval()
                self.logger.info('load checkpoints from'+load_best_path)

            dataloader = self.dev_data.dataloader
            my_prediction= []
            test_log_dict = OrderedDict()
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))
            metrics = {}

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
                    
                    all_input = self.prepare_data(batch)
                    output = self.model(**all_input)
                    test_loss = output[0]
                    self.add_logging(test_log_dict, 'loss', test_loss.item())
                    
                    decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                    generated_ids = self.model.generate(
                        inputs_embeds=all_input["inputs_embeds"],
                        attention_mask=all_input["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.args.max_output_length,
                        early_stopping=True
                    )
                    gen_text = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    gen_text = list(map(str.strip, gen_text))
                    my_prediction.extend(gen_text)

            if len(my_prediction) != 0:
                metrics = self.dev_data.evaluate(my_prediction, verbose=False)

            test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                        epoch=epoch, num_updates=num_updates, **metrics)
            if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
                self.model.train()

            for i,j in metrics.items():
                metric = i
                test_performance = j

            with open(self.args.output_dir + '/zero_result_dev.tsv', 'a') as fout:
                fout.write(str(test_performance)+'\n')

        return metrics
    
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

        valid_dataloader = self.dev_data.dataloader

        my_index = []
        my_prediction= []
        valid_log_dict = OrderedDict()
        self.logger.info("Begin validation on {:d} samples ...".format(len(self.dev_data.dataset)))
        metrics = {}

        with torch.no_grad():
            for local_step, batch in enumerate(valid_dataloader):
                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                valid_loss = output[0]
                self.add_logging(valid_log_dict, 'loss', valid_loss.item())
                
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                generated_ids = self.model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                my_prediction.extend(gen_text)
        if len(my_prediction) != 0:
            metrics = self.dev_data.evaluate(my_prediction, verbose=False)
        valid_loss = self.log_step(valid_log_dict, suffix="Valid |", tensorboard_suffix='valid',
                      epoch=epoch, num_updates=num_updates, **metrics)
        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
            self.model.train()
        return metrics

    def test(self, epoch=0, num_updates=0):
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

        with torch.no_grad():
            for local_step, batch in enumerate(test_dataloader):
                all_input = self.prepare_data(batch)
                output = self.model(**all_input)
                test_loss = output[0]
                self.add_logging(test_log_dict, 'loss', test_loss.item())
                
                decoder_input_ids = self.get_decoder_input_ids(all_input["inputs_embeds"])
                generated_ids = self.model.generate(
                    inputs_embeds=all_input["inputs_embeds"],
                    attention_mask=all_input["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.args.max_output_length,
                    early_stopping=True
                )
                gen_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                gen_text = list(map(str.strip, gen_text))
                my_prediction.extend(gen_text)
        if len(my_prediction) != 0:
            metrics = self.test_data.evaluate(my_prediction, verbose=False)
        test_loss = self.log_step(test_log_dict, suffix="Test |", tensorboard_suffix='test',
                      epoch=epoch, num_updates=num_updates, **metrics)
        if self.args.tune_method == 'model' or self.args.tune_method == 'lora' or self.args.tune_method == 'lora_stage2' or self.args.tune_method == 'bias' or self.args.tune_method == 'bias_stage2' or self.args.tune_method == 'adapter' or self.args.tune_method == 'prefix' or self.args.tune_method == 'hyper_PET':
            self.model.train()
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
                fout.write(task + ',' + str(float(test_loss))+'\n')

        return None

    def get_KL_divergence(self, target_task):
        import scipy.stats
        import numpy

        #load data
        dataloader = self.dev_data.dataloader
        self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

        #get target logits(use 64_tune ckpt)
        target_ckpt_path = self.args.random_tuned_ckpt_path + '/' + target_task + '/checkpoint-best.pt'

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
        from task_list import QA_task_list as TASK_LIST
        for task in TASK_LIST:
            source_checkpoint = self.args.source_checkpoint_path + '/' + task + '/checkpoint-best.pt'
            self.load_checkpoint(source_checkpoint)
            self.model.eval()

            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
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

            with open(self.args.output_dir+'/result.tsv', 'a') as fout:
                fout.write(task + ',' + str(KL_sum) + '\n')

        return None

    def get_block(self, target_task, mode):
        if mode == 'cos_withoutpad':
            for i in range(12):
                with open(self.args.output_dir+'/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open(self.args.output_dir+'/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')

            dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            #get target logits(use 64_tune ckpt)
            target_ckpt_path = self.args.random_tuned_ckpt_path + '/' + target_task + '/checkpoint-best.pt'

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
                source_checkpoint = self.args.source_checkpoint_path + '/' + task + '/checkpoint-best.pt'
                self.load_checkpoint(source_checkpoint)
                self.model.eval()

                with torch.no_grad():
                    for local_step, batch in enumerate(dataloader):
                        all_input = self.prepare_data(batch)
                        output = self.model(**all_input,output_hidden_states=True)
                        source_encoder_hidden_states = output['encoder_hidden_states']
                        source_decoder_hidden_states = output['decoder_hidden_states']
                        break

                cos_sum_list = [0 for i in range(24)]
                cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

                for i in range(len(target_attention_masks)):
                    if any(target_attention_masks[i]==0):
                        j = int(torch.nonzero(target_attention_masks[i]==0).squeeze(1)[0])
                        for block_num in range(12):
                            t_tensor = target_encoder_hidden_states[block_num][i,0:j,:]
                            s_tensor = source_encoder_hidden_states[block_num][i,0:j,:]
                            cos_sum_list[block_num] += cos_sim(t_tensor.view(-1),s_tensor.view(-1))
                    else:
                        for block_num in range(12):
                            t_tensor = target_encoder_hidden_states[block_num][i,:,:]
                            s_tensor = source_encoder_hidden_states[block_num][i,:,:]
                            cos_sum_list[block_num] += cos_sim(t_tensor.view(-1),s_tensor.view(-1))

                for i in range(len(target_labels)):
                    if any(target_labels[i]==-100):
                        j = int(torch.nonzero(target_labels[i]==-100).squeeze(1)[0])
                        for block_num in range(12):
                            t_tensor = target_decoder_hidden_states[block_num][i,0:j,:]
                            s_tensor = source_decoder_hidden_states[block_num][i,0:j,:]
                            cos_sum_list[block_num+12] += cos_sim(t_tensor.view(-1),s_tensor.view(-1))
                    else:
                        for block_num in range(12):
                            t_tensor = target_decoder_hidden_states[block_num][i,:,:]
                            s_tensor = source_decoder_hidden_states[block_num][i,:,:]
                            cos_sum_list[block_num+12] += cos_sim(t_tensor.view(-1),s_tensor.view(-1))

                for i in range(12):
                    with open(self.args.output_dir+'/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(cos_sum_list[i]))+'\n')
                    with open(self.args.output_dir+'/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(cos_sum_list[i+12]))+'\n')


    def get_EL2N(self):
        #load data
        dataloader = self.dev_data.dataloader
        self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

        #get source logits and cum KL_divergence
        from task_list import QA_task_list as TASK_LIST
        for task in TASK_LIST:
            source_checkpoint = self.args.source_checkpoint_path+'/'+task+'/checkpoint-best.pt'
            self.load_checkpoint(source_checkpoint)
            self.logger.info('load checkpoints from'+source_checkpoint)
            self.model.eval()

            source_logits_list = []
            source_labels_list = []
            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
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

            with open(self.args.output_dir+'/result.tsv', 'a') as fout:
                fout.write(task+','+str(float(EL2N_sum))+'\n')

        return None

    def get_logits_label_cos(self):
        #load data
        dataloader = self.dev_data.dataloader
        self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

        #get source logits and cum KL_divergence
        from task_list import QA_task_list as TASK_LIST
        for task in TASK_LIST:
            source_checkpoint = self.args.source_checkpoint_path+'/'+task+'/checkpoint-best.pt'
            self.load_checkpoint(source_checkpoint)
            self.logger.info('load checkpoints from'+source_checkpoint)
            self.model.eval()

            source_logits_list = []
            source_labels_list = []
            with torch.no_grad():
                for local_step, batch in enumerate(dataloader):
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

            with open(self.args.output_dir+'/result.tsv', 'a') as fout:
                fout.write(task+','+str(float(sim_sum))+'\n')

        return None
        
    
    def get_GraNd(self, target_task, mode):
        from math import sqrt

        if mode == 'layer':
            #单独计算每个layer的结果，没有叠加
            for i in range(12):
                with open(self.args.output_dir+'/encoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open(self.args.output_dir+'/encoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open(self.args.output_dir+'/decoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open(self.args.output_dir+'/decoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open(self.args.output_dir+'/decoder_block_'+str(i)+'_layer_2_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
            if target_task in ['boolq','mc_taco',]:
                cou_num = 3
            else:
                cou_num = 5 

            dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            from task_list import QA_task_list
            for task in QA_task_list:
                source_checkpoint = self.args.source_checkpoint_path+'/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_checkpoint)

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
                    with open(self.args.output_dir+'/encoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[2*i]))+'\n')
                    with open(self.args.output_dir+'/encoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[2*i+1]))+'\n')
                    with open(self.args.output_dir+'/decoder_block_'+str(i)+'_layer_0_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[24+3*i]))+'\n')
                    with open(self.args.output_dir+'/decoder_block_'+str(i)+'_layer_1_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[25+3*i]))+'\n')
                    with open(self.args.output_dir+'/decoder_block_'+str(i)+'_layer_2_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[26+3*i]))+'\n')
        
        elif mode == 'block':
            #单独计算每个block的结果，没有叠加
            for i in range(12):
                with open(self.args.output_dir+'/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
                with open(self.args.output_dir+'/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                    fout.write(target_task+'\n')
            if target_task in ['boolq','mc_taco',]:
                cou_num = 3
            else:
                cou_num = 5 

            dataloader = self.dev_data.dataloader
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            from task_list import QA_task_list
            for task in QA_task_list:
                source_checkpoint = self.args.source_checkpoint_path+'/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_checkpoint)

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
                    with open(self.args.output_dir+'/encoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[i]))+'\n')
                    with open(self.args.output_dir+'/decoder_block_'+str(i)+'_result.tsv', 'a') as fout:
                        fout.write(task+','+str(float(GraNd_sum_list[i+12]))+'\n')

        elif mode == 'dev':
            if target_task in ['boolq','mc_taco','amazon_polarity','tab_fact','scitail','tweet_eval-offensive',
                'tweet_eval-irony','glue-mrpc','glue-qqp','medical_questions_pairs','blimp-ellipsis_n_bar_1',
                'blimp-irregular_past_participle_adjectives','blimp-sentential_negation_npi_scope',]:
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
            self.logger.info("Begin test on {:d} samples ...".format(len(self.dev_data.dataset)))

            from task_list import QA_task_list as TASK_LIST
            for task in TASK_LIST:
                with open(self.args.output_dir+'/result.tsv', 'a') as fout:
                    fout.write(task + '\n')
                source_checkpoint = self.args.source_checkpoint_path+'/'+task+'/checkpoint-best.pt'
                self.load_checkpoint(source_checkpoint)

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
                
                with open(self.args.output_dir+'/result.tsv', 'a') as fout:
                    fout.write(task+','+str(float(GraNd_sum))+'\n')

        return None
        


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
        optimizer = Adafactor(params, lr=args.learning_rate, scale_parameter=False, relative_step=False, warmup_init=False)
        return optimizer


    def prepare_model_data(self, batch):
        all_input = {
            'input_ids': batch[0].to(self.device),
            'attention_mask': batch[1].to(self.device),
            'labels': batch[2].to(self.device),
            'decoder_attention_mask': batch[3].to(self.device),
        }
        input_ids = all_input.pop('input_ids')
        input_embeds = self.model.get_input_embeddings()(input_ids)
        all_input['inputs_embeds'] = input_embeds
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
        all_input['inputs_embeds'] = input_embeds
        return all_input

    def train_step(self, batch):
        all_input = self.prepare_data(batch)
        output = self.model(**all_input)
        loss = output[0] / self.gradient_accumulation_steps
        loss.backward()
        return loss
    
    def optimizer_step(self, parameters):
        updated = True
        scale = 0
        
        torch.nn.utils.clip_grad_norm_(parameters, self.args.max_grad_norm)
        self.optimizer.step()
        if updated:
            self.scheduler.step()
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
        for key, value in new_log_dict.items():
            if isinstance(value, float):
                message += ' {:s}: {:.10f}'.format(key, value)
            else:
                message += ' {:s}: {}'.format(key, value)
        self.logger.info(message)
        return new_log_dict.get('loss', None)
    
    def add_logging(self, log_dict, key, value):
        if key not in log_dict:
            log_dict[key] = []
        log_dict[key].append(value)
    
    def reset_logging(self, log_dict):
        for key in log_dict:
            log_dict[key] = []
