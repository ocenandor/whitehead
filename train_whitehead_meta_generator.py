import argparse
import json
from copy import copy
from functools import partial
from os import environ
from pathlib import Path
from pickle import load

import numpy as np
import torch
import wandb
from configs.utils import get_config
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, RunningAverage
from lming.optimization import build_optimizer, build_scheduler
from lming.utils import checkpoints_path, download_artifact
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          PreTrainedTokenizerFast)
from utils import (calculate_nan_number, completion_ratio_v2_1pos_match,
                   completion_ratio_v2_2pos_match,
                   completion_ratio_v2_any_pos_match,
                   completion_ratio_v2_full_match, get_slice,
                   inference_collate_fn, str_prefix2bool, train_collate_fn)

from freegroup.tools import batch_magnus_is_from_normal_closure, from_string
#TODO REWRITE AS TRAIN FUNCTION FOR HYPER SEARCH
# WANDB SETTINGS
with open("env.json") as f:
    envs = json.load(f)
environ.update(envs)
environ["WANDB_JOB_TYPE"] = "train"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', type=str, required=False, default='configs/config_whitehead.json', help='config path')
args = parser.parse_args()

# TECHNICAL STAFF
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
config = get_config(args.cfg, device=device)

torch.set_default_dtype(eval(f"torch.{config['dtype']}"))
torch.manual_seed(seed=config['train_seed'])
torch.set_float32_matmul_precision('high')

# INIT WANDB
wandb.init(config=config)
config = wandb.run.config

# INIT MODEL
model_config_dir = download_artifact(config['model_name'])
print(f'Using model config from {model_config_dir}')
model_config = AutoConfig.from_pretrained(model_config_dir)
print(f'Using model config: {model_config}')
model = AutoModelForCausalLM.from_config(model_config).to(device)
model = torch.compile(model)
learnable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

# INIT TOKENIZER
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config_dir)

# DOWNLOAD DATA
## TRAIN
data = download_artifact(config.dataset_name)
print('download data from', Path(data))
with open(Path(data) / config.train['dataset_filename'], 'rb') as file:
    train_dataset = get_slice(load(file), config.train['dataset_size'])
# TEST 
with open(Path(data) / config.inference['dataset_filename'], 'rb') as file:
    test_dataset_tmp = load(file)

test_datasets = {
    'low_info': get_slice(list(filter(lambda el: el.get('test_2', None) == 'low_info', test_dataset_tmp)), config.inference['dataset_size']),
    'medium_info': get_slice(list(filter(lambda el: el.get('test_2', None) == 'medium_info', test_dataset_tmp)), config.inference['dataset_size']),
    'zero_info': get_slice(list(filter(lambda el: el.get('test_2', None) is None, test_dataset_tmp)), config.inference['dataset_size']),
}
for k, v in test_datasets.items():
    print(f'Using test {k} with size {len(v)}')

## VALIDATION
val_size = config.validation['dataset_size']
validation_dataset = get_slice(train_dataset, val_size)
train_dataset = train_dataset[len(validation_dataset):] 
print(f'Train Dataset of size: {len(train_dataset)}. Validation Dataset of size: {len(validation_dataset)}')

train_loader = DataLoader(
    train_dataset, batch_size=config.train['batch_size'],
    shuffle=True, num_workers=1, collate_fn=partial(train_collate_fn, tokenizer=tokenizer, key='prompt')
    )
validation_loader = DataLoader(
    validation_dataset, batch_size=config.validation['batch_size'],
    shuffle=False, collate_fn=partial(train_collate_fn, tokenizer=tokenizer, key='prompt')
    )
PREFIX_LENGTH_LIST = [0, 3, 10]
inference_dataloaders = {}
for name, test_dataset in test_datasets.items():
    for prefix_length in PREFIX_LENGTH_LIST:
        inference_loader = DataLoader(
                list(filter(lambda x: x['word_length'] >= prefix_length, test_dataset)),
                batch_size=config.inference['batch_size'], num_workers=1,shuffle=False,
                collate_fn=partial(inference_collate_fn, tokenizer=tokenizer, prefix_length=prefix_length, label='yy')
            ) 
        # inference_loader.test_name = name
        inference_dataloaders[(name, prefix_length)] = inference_loader

print(f'Using: {len(PREFIX_LENGTH_LIST)}x{len(test_datasets)} (prefix_lengths x tests) inferencers.')


# OPTIMIZER
optimizer_config = copy(config.optimizer)
optimizer = build_optimizer(learnable_parameters, **optimizer_config)

scheduler_config = copy(config.scheduler)
print('Using scheduler config:', scheduler_config)
scheduler_use = scheduler_config.pop('use')
scheduler_update_every = scheduler_config.pop('update_every')
scheduler_warmup_duration = scheduler_config.pop('warmup_duration')
scheduler_decay_duration = scheduler_config.pop('decay_duration')
scheduler = build_scheduler(optimizer,
    warmup_duration = scheduler_warmup_duration // scheduler_update_every,
    decay_duration = scheduler_decay_duration // scheduler_update_every,
    **scheduler_config
)

gradient_clipping_config = copy(config.gradient_clipping)
gradient_clipping_use = gradient_clipping_config.pop('use')

# IGNITE FUNCS
## TRAIN FUNC
def train(engine, batch):
    model.train()
    result = model(**batch.to(device))
    loss = result['loss']

    optimizer.zero_grad()
    loss.backward() 
    if gradient_clipping_use:
        torch.nn.utils.clip_grad_norm_(learnable_parameters, **gradient_clipping_config)
    optimizer.step()

    return {
        'loss': loss.item(),
        'y_pred': result['logits'][:, :-1].detach().cpu(),
        'y_true': batch['labels'][:, 1:].detach().cpu(),
    }
## VALIDATION FUNC
def validation(engine, batch):
    model.eval()
    with torch.no_grad():
        result = model(**batch.to(device))
        loss = result['loss']

    return {
        'loss': loss.item(),
        'y_pred': result['logits'][:, :-1].detach().cpu(),
        'y_true': batch['labels'][:, 1:].detach().cpu(),
    }

## TEST FUNC
generation_config = config['generation']
generation_config['suppress_tokens'] = tokenizer.convert_tokens_to_ids(generation_config['suppress_tokens'])

def inference(engine, batch, prefix_length=0, n_proc=8):
    model.eval()
    start_idx = batch['input_ids'].shape[1]
    with torch.no_grad():
        result = model.generate(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device),
                                **generation_config)
        decodes = tokenizer.batch_decode(result[:, start_idx-prefix_length:], skip_special_tokens=True)
        decodes = list(map(from_string, decodes))

    ###calculate middle results #TODO put into another separate function###=======================
    #cycle_1
    rels_1 = list(map(lambda x: x[0], batch['relations']))
    completion_ratio_1 = batch_magnus_is_from_normal_closure(decodes, rels_1, n_proc=n_proc, timelimit=0.1)
    #cycle_2
    rels_2 = list(map(lambda x: x[1], batch['relations']))
    completion_ratio_2 = batch_magnus_is_from_normal_closure(decodes, rels_2, n_proc=n_proc, timelimit=0.1)

    preds = (np.array([completion_ratio_1, completion_ratio_2]).T)
    gts = (np.array(list(map(str_prefix2bool, batch['prefixes']))))

    return {
        'preds': preds,
        'gts': gts
    }

# IGNITE ENGINES
trainer = Engine(train)
validator = Engine(validation)
inferencers = {key: Engine(partial(inference, prefix_length=key[1])) for key in inference_dataloaders}

def log_wandb(engine, prefix):
    ''' Log all metrics from engine state '''
    wandb.run.log({f'{prefix}/{k}': v for k, v in engine.state.metrics.items()}, step=trainer.state.iteration)

# IGNITE METRICS & WANDB LOGGING

## TRAIN METRICS
RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'running_loss')
## LOGGING
trainer.add_event_handler(
    Events.ITERATION_COMPLETED(every=config.train['log_every']),
    partial(log_wandb, prefix=config.train['log_prefix']),
)

ProgressBar().attach(trainer, ['running_loss'])

## VALIDATION METRICS
RunningAverage(output_transform=lambda x: x['loss']).attach(validator, 'running_loss')
## LOGGING
validator.add_event_handler(
    Events.ITERATION_COMPLETED(every=config.validation['log_every']),
    partial(log_wandb, prefix=config.validation['log_prefix']),
)

ProgressBar().attach(validator, ['running_loss'])

## TEST METRICS
for (test_name, prefix_length), inferencer in inferencers.items():
    Average(output_transform=completion_ratio_v2_full_match).attach(inferencer, f'cr_fm')
    Average(output_transform=completion_ratio_v2_1pos_match).attach(inferencer, f'cr_1p')
    Average(output_transform=completion_ratio_v2_2pos_match).attach(inferencer, f'cr_2p')
    # Average(output_transform=completion_ratio_v2_any_pos_match).attach(inferencer, f'cr_ap_pr={prefix_length}')
    Average(output_transform=calculate_nan_number).attach(inferencer, f'nans')

    ### LOGGING
    inferencer.add_event_handler(Events.COMPLETED,
                                 partial(log_wandb,
                                         prefix=config.inference['log_prefix']+ f'_{test_name}_pr={prefix_length}'
                                         )
                                )
    ProgressBar().attach(inferencer)


to_save = {'model': model}
best_checkpoint = ModelCheckpoint(dirname=checkpoints_path() / config.inference['save_dirname'],
                                  score_name='cr_fm', n_saved=1,
                                  global_step_transform=lambda *_: trainer.state.iteration)
inferencers[('zero_info', 3)].add_event_handler(Events.COMPLETED, best_checkpoint, to_save)

last_checkpoint = ModelCheckpoint(dirname=checkpoints_path() / config.inference['save_dirname'],
                                  n_saved=15,
                                  global_step_transform=lambda *_: trainer.state.iteration)
trainer.add_event_handler(Events.EPOCH_COMPLETED(every=config.train['max_epochs'] // 15), last_checkpoint, to_save)
trainer.add_event_handler(Events.COMPLETED, last_checkpoint, to_save)

# IGNITE SCHEDULER
if scheduler_use:
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=scheduler_update_every),
                              lambda: scheduler.step())
    
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every = config.train['log_every']),
                              lambda: wandb.log({'learning_rate': scheduler.get_last_lr()[0]},
                                                step=trainer.state.iteration)
)

# RUN VALIDATION
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    # Events.ITERATION_COMPLETED(every=config.validation['every']),
    lambda: validator.run(validation_loader)
)

# RUN INFERENCE
def run_inference():
    # number of inferencers = number of prefix length variants
    for key, inferencer in inferencers.items(): #prefix length is a property of inferencer (in decode), #test type - is a property of both dataloader(collate) and inferencer (for logging)
        inferencer.run(inference_dataloaders[key])

trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    # Events.ITERATION_COMPLETED(every=config.inference['every']),
    run_inference
)

# START TRAINING
run_inference()
trainer.run(train_loader,
            # epoch_length=config.train['epoch_length'],
            max_epochs=config.train['max_epochs'])


wandb.finish()
