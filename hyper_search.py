import argparse
import json
import os
from copy import copy
from functools import partial
from os import environ
from pathlib import Path
from pickle import load

import numpy as np
import torch
import torch.nn as nn
import wandb
from configs.utils import (get_config, selective_merge,  # , load_config
                           set_default_config)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import (EMAHandler, ModelCheckpoint,
                             create_lr_scheduler_with_warmup)
from ignite.metrics import (Accuracy, Average, MetricsLambda, MetricUsage,
                            RunningAverage)
from lming.optimization import build_optimizer, build_scheduler
from lming.utils import (checkpoints_path, download_artifact, from_tensor,
                         to_tensor)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          PreTrainedTokenizerFast)
from utils import (PosNegCrossEntropy, back_transform, get_slice,
                   inference_collate_fn, load_external_module,
                   max_gamma_metric, train_collate_fn,
                   transform_completion_ratio, transform_reduction_ratio)


# WANDB SETTINGS
with open("env.json") as f:
    envs = json.load(f)
environ.update(envs)
environ["WANDB_JOB_TYPE"] = "train"

# TECHNICAL STAFF
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision('high')

def train(config=None):
    # parser = argparse.ArgumentParser()
    # args = parser.add_argument('-c', '--cfg', type=str, required=False, default='configs/config_whitehead.json', help='config path')
    config = get_config('configs/config_whitehead.json', device=device)

    torch.set_default_dtype(eval(f"torch.{config['dtype']}"))
    torch.manual_seed(seed=config['train_seed'])

    with wandb.init(config=config):
        config = wandb.run.config

        # INIT MODEL
        model_config_dir = download_artifact(config['model_name'])
        print(f'Using model config from {model_config_dir}')
        model_config = AutoConfig.from_pretrained(model_config_dir)
        print(f'Using model config: {model_config}')
        model = AutoModelForCausalLM.from_config(model_config).to(device)
        model = torch.compile(model)

        # INIT TOKENIZER
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config_dir)

        # DOWNLOAD DATA
        ## TRAIN
        data = download_artifact(config.dataset_name)
        print('download data from', Path(data))
        with open(Path(data) / config.train['dataset_filename'], 'rb') as file:
            train_dataset = get_slice(load(file), config.train['dataset_size'])
        ## TEST
        with open(Path(data) / config.inference['dataset_filename'], 'rb') as file:
            test_dataset = get_slice(load(file), config.inference['dataset_size'])
        print(f'Using test with size {len(test_dataset)}')
        ## VALIDATION
        val_size = config.validation['dataset_size']
        validation_dataset = get_slice(train_dataset, val_size)
        train_dataset = train_dataset[len(validation_dataset):] 
        print(f'Train Dataset of size: {len(train_dataset)}. Validation Dataset of size: {len(validation_dataset)}')

        train_loader = DataLoader(
            train_dataset, batch_size=config.train['batch_size'],
            shuffle=True, num_workers=1, collate_fn=partial(train_collate_fn, tokenizer=tokenizer)
            )
        validation_loader = DataLoader(
            validation_dataset, batch_size=config.validation['batch_size'],
            shuffle=False, collate_fn=partial(train_collate_fn, tokenizer=tokenizer)
            )
        inference_loader = DataLoader(
            test_dataset, batch_size=config.inference['batch_size'], num_workers=1,
            shuffle=False, collate_fn=partial(inference_collate_fn, tokenizer=tokenizer)
            )

        learnable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

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

        def inference(engine, batch):
            model.eval()
            n = generation_config['num_return_sequences']
            with torch.no_grad():
                closures = batch.pop('closures')
                result = model.generate(**batch.to(device), **generation_config)
                decodes = from_tensor(result.cpu(), tokenizer=tokenizer)

            return {
                'outputs': decodes,
                'closures': [[el for el in subclosure for _ in range(n)] for subclosure in closures]
            }
        # IGNITE ENGINES
        trainer = Engine(train)
        validator = Engine(validation)
        inferencer = Engine(inference)

        def log_wandb(engine, prefix):
            ''' Log all metrics from engine state '''
            wandb.run.log({f'{prefix}/{k}': v for k, v in engine.state.metrics.items()}, step=trainer.state.iteration)

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
        average_completion_ratio = Average(
            output_transform=partial(transform_completion_ratio,
                                    fdim=config.fdim,
                                    max_shots=config.inference['max_shots'],
                                    union=False)
            )

        average_reduction_ratio = Average(
            output_transform=partial(transform_reduction_ratio,
                                    fdim=config.fdim,
                                    max_shots=config.inference['max_shots'])
        )
        idx = 4 # completion ratio by (idx + 1) generated prompts
        MetricsLambda(partial(back_transform, idx=idx), average_completion_ratio).attach(inferencer, f'completion_ratio_intersection_{idx + 1}')
        # MetricsLambda(partial(back_transform, idx=idx), average_completion_ratio_union).attach(inferencer, f'completion_ratio_union_{idx + 1}')
        MetricsLambda(partial(back_transform, idx=idx), average_reduction_ratio).attach(inferencer, f'reduction_ratio_{idx + 1}')
        Average(output_transform=partial(max_gamma_metric, fdim=config.fdim, target_gamma=2, save_pos_examples=True, n_proc=10)).attach(inferencer, f'max_gamma_contains_{2}')
        ## LOGGING
        inferencer.add_event_handler(Events.COMPLETED, partial(log_wandb, prefix=config.inference['log_prefix']))
        if config.inference['save_best']:
            for score_name in [config.inference['score_name'], "max_gamma_contains_2"]:
                best_checkpoint = ModelCheckpoint(
                    dirname=checkpoints_path() / config.inference['save_dirname'],
                    score_name=score_name, n_saved=2,
                    global_step_transform=lambda *_: trainer.state.iteration,
                )
                inferencer.add_event_handler(Events.COMPLETED, best_checkpoint, {
                    'model': model,
                    })
            
        ProgressBar().attach(inferencer, ['completion_ratio_intersection_5'])

        # IGNITE SCHEDULER
        if scheduler_use:
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=scheduler_update_every),
                                    lambda: scheduler.step())
            
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every = config.train['log_every']),
                                    lambda: wandb.log({'learning_rate': scheduler.get_last_lr()[0]},
                                                        step=trainer.state.iteration)
        )

        # RUN INFERENCE
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            # Events.ITERATION_COMPLETED(every=config.inference['every']),
            lambda: inferencer.run(inference_loader)
        )

        # START TRAINING
        trainer.run(train_loader,
                    # epoch_length=config.train['epoch_length'],
                    max_epochs=config.train['max_epochs'])

        try:
            # START TRAINING
            trainer.run(train_loader,
                        # epoch_length=config.train['epoch_length'],
                        max_epochs=config.train['max_epochs'])
        except KeyboardInterrupt:
            torch.cuda.empty_cache()
            print('Stopped by the user')
        except BaseException:
            import traceback
            print(traceback.format_exc())
            exit_code = 1


if __name__ == '__main__':
    with open('configs/sweep_lr_warmup_config.json') as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(sweep_config,
                           project="whitehead", entity='ml-in-algebraic-topology')
    wandb.agent(sweep_id, train)