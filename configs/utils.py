import json

import numpy as np


def set_default(d, key, default):
    d[key] = d.get(key, default)

def set_default_config(config):

    # general
    set_default(config, 'verbose', 1)
    set_default(config, 'train_seed', 0)
    set_default(config, 'dtype', 'float32')
    set_default(config, 'device', 'cpu')
    set_default(config, 'fdim', 3)
    set_default(config, 'model_name', 'gpt-neo-x-fdim-3:v2')
    set_default(config, 'nb_weights', 'gpt-neo-x-fdim-3-NB:v0')
    set_default(config, 'dataset_name', 'fdim-3:v3')

    # train config
    train_config = config.get('train', dict())
    set_default(train_config, 'log_prefix', 'train')
    set_default(train_config, 'log_every', 1_000)
    set_default(train_config, 'dataset_filename', 'train.pkl')
    set_default(train_config, 'dataset_size', None)
    set_default(train_config, 'nb_alpha', 0.5)
    set_default(train_config, 'max_epochs', 10 ** 4)
    set_default(train_config, 'epoch_length', None)
    set_default(train_config, 'batch_size', 2 ** 5)
    set_default(train_config, 'save_milestones',
            np.linspace(0, 1_000_000, 10, endpoint=False, dtype=int).tolist())
    set_default(train_config, 'save_last', True)
    set_default(train_config, 'save_last_every', 10_000)
    config['train'] = train_config

    # optimizer config
    optimizer_config = config.get('optimizer', dict())
    set_default(optimizer_config, 'name', 'AdamW')
    set_default(optimizer_config, 'lr', 1e-6)
    config['optimizer'] = optimizer_config

    # scheduler config
    scheduler_config = config.get('scheduler', dict())
    set_default(scheduler_config, 'use', True)
    set_default(scheduler_config, 'update_every', 10)
    set_default(scheduler_config, 'start_lr', optimizer_config['lr'])
    end_factor = scheduler_config.pop('end_factor', 0.1)
    set_default(scheduler_config, 'end_lr', end_factor * scheduler_config['start_lr'])
    set_default(scheduler_config, 'warmup_duration', 20_000)
    set_default(scheduler_config, 'decay_duration', 200_000)
    set_default(scheduler_config, 'name', 'LinearLR')
    config['scheduler'] = scheduler_config

    # gradient clipping config
    gradient_clipping_config = config.get('gradient_clipping', dict())
    set_default(gradient_clipping_config, 'use', True)
    set_default(gradient_clipping_config, 'max_norm', 1.)
    config['gradient_clipping'] = gradient_clipping_config

    # validation config
    validation_config = config.get('validation', dict())
    set_default(validation_config, 'log_prefix', 'val')
    set_default(validation_config, 'every', 1_000)
    set_default(validation_config, 'dataset_size', 0.1)
    set_default(validation_config, 'batch_size', 2 ** 4)
    set_default(validation_config, 'epoch_length', 50)
    config['validation'] = validation_config
    val_size = config['validation']['dataset_size']
    assert isinstance(val_size, int) or isinstance(val_size, float)

    # inference config
    inference_config = config.get('inference', dict(test = {}))
    for k, v in inference_config.items():
        set_default(v, 'log_prefix', k)
        set_default(v, 'dataset_filename', f'{k}.pkl')
        set_default(v, 'dataset_size', None)
        set_default(v, 'every', 2_000)
        set_default(v, 'batch_size', 2 ** 4)
        set_default(v, 'epoch_length', None)
        set_default(v, 'max_shots', 5)
        set_default(v, 'save_best', False)
        set_default(v, 'save_dirname', k)
        set_default(v, 'score_name', 'completion_ratio_5')
        set_default(v, 'maximize', True)
        set_default(v, 'n_saved', 1)
    config['inference'] = inference_config

    # generation config
    generation_config = config.get('generation', dict())
    set_default(generation_config, 'max_length', 120)
    set_default(generation_config, 'suppress_tokens', ['[', ']', ',', 'y', 'n', ':', '<s>'])
    set_default(generation_config, 'num_return_sequences', 5)
    set_default(generation_config, 'do_sample', True)
    config['generation'] = generation_config
    print(f"Using generation config: {generation_config}")

    #ema config #TODO what is EMA
    ema_config = config.get('ema', dict())
    set_default(ema_config, 'use', True)
    set_default(ema_config, 'update_every', 5)
    config['ema'] = ema_config

def selective_merge(base_obj, delta_obj):
    if not isinstance(base_obj, dict):
        return delta_obj
    common_keys = set(base_obj).intersection(delta_obj)
    new_keys = set(delta_obj).difference(common_keys)
    for k in common_keys:
        base_obj[k] = selective_merge(base_obj[k], delta_obj[k])
    for k in new_keys:
        base_obj[k] = delta_obj[k]
    return base_obj

def get_config(user_config, default_config='configs/default_config.json', *args, device='cpu'):
    with open(default_config) as f:
        base_config = json.load(f)
    with open(user_config) as f:
        config = json.load(f)
        config['device'] = device
    config = selective_merge(base_config, config)
    return config

if __name__ == '__main__':
    config = {}
    set_default_config(config)
    with open('default_config.json', 'w') as f:
        json.dump(config, f, indent=2)