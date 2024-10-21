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
from configs.utils import load_config, selective_merge, set_default_config
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import (EMAHandler, ModelCheckpoint,
                             create_lr_scheduler_with_warmup)
from ignite.metrics import (Accuracy, Average, MetricsLambda, MetricUsage,
                            RunningAverage)
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          PreTrainedTokenizerFast)
from utils import (back_transform, get_slice, inference_collate_fn,
                   load_external_module, train_collate_fn,
                   transform_completion_ratio, transform_reduction_ratio, PosNegCrossEntropy)

load_external_module('lming', '/main/draft-v2/lming/__init__.py')
load_external_module('tokenizer', '/main/draft-v2/tokenizer.py')


from lming.optimization import build_optimizer, build_scheduler
from lming.utils import (checkpoints_path, download_artifact, from_tensor,
                         to_tensor)


def train(config=None):
    config = load_config('configs/config.json', device='cuda' if torch.cuda.is_available() else 'cpu')

    with wandb.init(config=config):
        config = wandb.config
        torch.set_default_dtype({'float32': torch.float32, 'float16': torch.float16}[config.dtype])
        torch.manual_seed(seed = config.train_seed)

        ### load model
        model_config_dir = download_artifact(config.model_name)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config_dir)
        model_config = AutoConfig.from_pretrained(model_config_dir)

        nb_weights_dir = download_artifact(config.nb_weights)
        nb_weights_name = os.listdir(nb_weights_dir)[0]
        nb_weights = torch.load(nb_weights_dir + '/' + nb_weights_name, map_location=config.device)
        model = AutoModelForCausalLM.from_config(model_config).to(config.device)

        nb_model = AutoModelForCausalLM.from_config(model_config).to(config.device)
        nb_model.load_state_dict(nb_weights)
        nb_model.eval()

        ema_config = copy(config.ema)
        ema_use = ema_config.pop('use')
        ema_update_every = ema_config.pop('update_every')

        if ema_use:
            ema = EMAHandler(model, **ema_config)
            test_model = ema.ema_model
        else:
            test_model = model

        ### load data
        data = download_artifact(config.dataset_name)
        with open(Path(data) / config.train['dataset_filename'], 'rb') as file:
            train_dataset = get_slice(load(file), config.train['dataset_size'])

        # download (multiple) test data
        test_datasets = {}
        for k, v in config.inference.items():
            with open(Path(data) / v['dataset_filename'], 'rb') as file:
                test_datasets[k] = get_slice(load(file), v['dataset_size'])
                if config.verbose > 1: print(f'Using {k} with size {len(test_datasets[k])}')

        val_size = config.validation['dataset_size']
        validation_dataset = get_slice(train_dataset, val_size)
        train_dataset = train_dataset[len(validation_dataset):]

        if config.verbose >= 1:
            print(f'Train Dataset of size: {len(train_dataset)}. Validation Dataset of size: {len(validation_dataset)}')

        ### Dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size = config.train['batch_size'], shuffle = False, num_workers = 1,
            collate_fn = partial(train_collate_fn, tokenizer = tokenizer, fdim = config.fdim)
            )

        validation_loader = DataLoader(
            validation_dataset, batch_size = config.validation['batch_size'], shuffle = True,
            collate_fn = partial(train_collate_fn, tokenizer = tokenizer, fdim = config.fdim)
        )

        inference_loaders = {
            k: DataLoader(test_datasets[k], batch_size = v['batch_size'], shuffle = True,
                        collate_fn = partial(inference_collate_fn, tokenizer = tokenizer, fdim = config.fdim))
            for k, v in config.inference.items()
        }

        ### OPtimizer
        learnable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer_config = copy(config.optimizer)
        optimizer = build_optimizer(model.parameters(), **optimizer_config)
        scheduler_config = copy(config.scheduler)
        scheduler_use = scheduler_config.pop('use')
        scheduler_update_every = scheduler_config.pop('update_every')
        scheduler_warmup_duration = scheduler_config.pop('warmup_duration')
        scheduler_decay_duration = scheduler_config.pop('decay_duration')
        scheduler = build_scheduler(optimizer,
            warmup_duration = scheduler_warmup_duration // scheduler_update_every,
            decay_duration = scheduler_decay_duration // scheduler_update_every,
            **scheduler_config
        )

        criteria = PosNegCrossEntropy(config, nb_model)
        
        if config.verbose > 1: 
            print(f'Using model with: {sum(p.numel() for p in model.parameters()) / (10 ** 6)}M parameters')
        gradient_clipping_config = copy(config.gradient_clipping)
        gradient_clipping_use = gradient_clipping_config.pop('use')

        generation_config = copy(config.generation)
        generation_config['suppress_tokens'] = tokenizer.convert_tokens_to_ids(generation_config['suppress_tokens'])


        def train(engine, batch):
                
            for k, v in batch.items():
                batch[k] = v.to(config.device)
            
            model.train()

            outputs = model(**batch)
            loss = criteria(outputs, batch)

            optimizer.zero_grad()
            # outputs['loss'].backward()
            loss.backward() # TODO Custom loss
            if gradient_clipping_use:
                torch.nn.utils.clip_grad_norm_(learnable_parameters, **gradient_clipping_config)
                
            optimizer.step()

            return {
                'loss': loss.item(),
                'y_pred': outputs['logits'][:, :-1].detach().cpu(),
                'y_true': batch['labels'][:, 1:].detach().cpu(),
            }

        def validation(engine, batch):
            
            for k, v in batch.items():
                batch[k] = v.to(config.device)
            
            test_model.eval()
            with torch.no_grad():
                outputs = test_model(**batch)
            loss = criteria(outputs, batch)
            return {
                'loss': loss.item(),
                'y_pred': outputs['logits'][:, :-1].detach().cpu(),
                'y_true': batch['labels'][:, 1:].detach().cpu(),
            }

        def inference(engine, batch):
            batch['inputs'] = batch['inputs'].to(config.device)    
            
            test_model.eval()
            outputs = test_model.generate(inputs = batch['inputs'], **generation_config)

            return {
                'prefixes': batch['prefixes'],
                'outputs': from_tensor(outputs.cpu(), tokenizer = tokenizer)
            }
        
        trainer = Engine(train)
        validator = Engine(validation)
        inferencers = {k: Engine(inference) for k in config.inference.keys() }


        RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'running_loss')
        Average(output_transform=lambda x: x['loss']).attach(trainer, 'loss', MetricUsage(
            started = Events.EPOCH_STARTED | Events.ITERATION_STARTED(every = config.train['log_every'] + 1),
            iteration_completed = Events.ITERATION_COMPLETED,
            completed = Events.ITERATION_COMPLETED(every = config.train['log_every']))
        )
        Accuracy(
            output_transform=lambda x: (x['y_pred'].reshape(-1, len(tokenizer.get_vocab())) , x['y_true'].reshape(-1))
        ).attach(trainer, 'accuracy', MetricUsage(
            started = Events.STARTED | Events.ITERATION_STARTED(every = config.train['log_every'] + 1),
            iteration_completed = Events.ITERATION_COMPLETED,
            completed = Events.ITERATION_COMPLETED(every = config.train['log_every']))
        )

        Average(output_transform=lambda x: x['loss']).attach(validator, 'loss')
        Accuracy(
            output_transform=lambda x: (x['y_pred'].reshape(-1, len(tokenizer.get_vocab())) , x['y_true'].reshape(-1))
        ).attach(validator, 'accuracy')

        for inferencer in inferencers.values():
            average_completion_ratio = Average(
                output_transform=partial(transform_completion_ratio, fdim = config.fdim, max_shots = v['max_shots'])
            )
            average_reduction_ratio = Average(
                output_transform=partial(transform_reduction_ratio, fdim = config.fdim, max_shots = v['max_shots'])
            )
            for idx in range(0, v['max_shots']):
                MetricsLambda(partial(back_transform, idx = idx), average_completion_ratio).attach(inferencer, f'completion_ratio_{idx + 1}')
                MetricsLambda(partial(back_transform, idx = idx), average_reduction_ratio).attach(inferencer, f'reduction_ratio_{idx + 1}')

        if scheduler_use:
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=scheduler_update_every), lambda: scheduler.step())
        if ema_use:
            ema.attach(trainer, 'ema', Events.ITERATION_COMPLETED(every=ema_update_every))


        def log_wandb(engine, prefix):
            wandb.run.log({f'{prefix}/{k}': v for k, v in engine.state.metrics.items()}, step = trainer.state.iteration)

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every = config.train['log_every']),
            partial(log_wandb, prefix = config.train['log_prefix']),
        )

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every = config.train['log_every']),
            lambda: wandb.log({'learning_rate': scheduler.get_last_lr()[0]}, step = trainer.state.iteration)
        )

        ProgressBar().attach(trainer, ['running_loss'])

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every = config.validation['every']),
            lambda: validator.run(validation_loader, epoch_length = config.validation['epoch_length'])
        )

        validator.add_event_handler(
            Events.COMPLETED, partial(log_wandb, prefix = config.validation['log_prefix'])
        )

        for k, v in config.inference.items():
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every = v['every']),
                lambda: inferencers[k].run(inference_loaders[k], epoch_length = v['epoch_length'])
            )
            inferencers[k].add_event_handler(Events.COMPLETED, partial(log_wandb, prefix = v['log_prefix']))
            if v['save_best']:
                best_checkpoint = ModelCheckpoint(
                    dirname = checkpoints_path() / v['save_dirname'],
                    score_name = v['score_name'],
                    global_step_transform = lambda *_: trainer.state.iteration,
                )
                inferencers[k].add_event_handler(Events.COMPLETED, best_checkpoint, {'model': test_model})
 

        to_save_train = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'trainer': trainer}
        if ema_use:
            to_save_train['ema'] = ema.ema_model

        if isinstance(config.train['save_milestones'], list):
            milestone_checkpoint = ModelCheckpoint(
                dirname = checkpoints_path() / 'milestones',
                n_saved = len(config.train['save_milestones']),
                global_step_transform = lambda *_: trainer.state.iteration,
            )
            trainer.add_event_handler(Events.ITERATION_COMPLETED(event_filter=lambda *_: trainer.state.iteration in config.train['save_milestones']),
                                    milestone_checkpoint, to_save_train)

        if config.train['save_last']:
            last_checkpoint = ModelCheckpoint(
                dirname = checkpoints_path() / 'last',
                global_step_transform = lambda *_: trainer.state.iteration,
            )
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every = config.train['save_last_every']),
                                    last_checkpoint, to_save_train)
            

        exit_code = 0

        try:
            trainer.run(train_loader, epoch_length = config.train['epoch_length'], max_epochs = config.train['max_epochs'])
        except KeyboardInterrupt:
            print('Stopped by the user')
        except BaseException:
            import traceback
            print(traceback.format_exc())
            exit_code = 1



if __name__ == '__main__':
    with open('configs/sweep_nb_alpha_config.json') as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(sweep_config,
                           project="lming", entity='ml-in-algebraic-topology')
    wandb.agent(sweep_id, train, count=14)