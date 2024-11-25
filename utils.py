import importlib.util
import math
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from freegroup.derivatives import max_gamma_contains
from freegroup.tools import normalize


def load_external_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

# load_external_module('lming', '/main/draft-v2/lming/__init__.py')

from lming.metrics import (batch_few_shot_completion_ratio,
                           batch_few_shot_reduction_ratio)
from lming.utils import to_tensor


def get_slice(array, slice):
    array = [el for el in array]
    if slice is None: return array
    if isinstance(slice, int): return array[:slice]
    if isinstance(slice, float): return array[:int(len(array) * slice)]
    assert False, "Unknown `slice` type"

def transform_completion_ratio(x, fdim=3, max_shots=5, union=False):
    outputs = batch_few_shot_completion_ratio(x['outputs'], fdim=fdim, max_shots=max_shots, closures=x['closures'], union=union)
    return torch.tensor(list(outputs.values())).reshape(1, -1)

def transform_reduction_ratio(x, fdim=3, max_shots=5):
    outputs = batch_few_shot_reduction_ratio(x['outputs'], fdim=fdim, max_shots=max_shots, closures=x['closures'])
    outputs = {k: v for k, v in outputs.items() if k.startswith('mean')}
    return torch.tensor(list(outputs.values())).reshape(1, -1)

def back_transform(x, idx):
    return x[idx].item()

def inverse(arr):
    arr = np.array(arr)
    arr = arr * (-1)
    return list(arr)[::-1]

def max_gamma_metric(x, fdim=3, target_gamma=4, save_pos_examples=False, n_proc=None):    
    # nontrivial = [-2, -1, 2, 1, -3, -2, -1, 2, 3, -2, 1, 2, -1, -3, -2, 1, 2, 3]
    # res = np.array([max_gamma_contains([normalize(inverse(nontrivial) + w)], fdim=fdim)[0] for w in x['outputs']])
    if n_proc is not None and n_proc > 1:
        with mp.Pool(n_proc) as pool:
            results = pool.starmap(max_gamma_contains, [([normalize(w)], fdim, False) for w in x['outputs']])
    else:
        results = [max_gamma_contains([normalize(w)], fdim, keepdim=False) for w in x['outputs']]

    res = np.array(results)
    if save_pos_examples:
        save_dir = './training_generated_words'
        Path(save_dir).mkdir(parents=False, exist_ok=True)
        files = sorted(os.listdir(save_dir))
        if len(files) >= 1000:
            os.remove(save_dir + '/' + files[0])
        if len(files) == 0:
            idx = 0
        else:
            idx = int(files[-1].split('/')[-1].split('.')[0]) + 1 # Number of saved batches
        samples = {gamma: [x['outputs'][idx] for idx in np.where(res == gamma)[0]] for gamma in set(list(res))}
        with open(save_dir + f'/{idx:010d}.pkl', 'wb') as f:
            pickle.dump(samples, f)

    res = (res == target_gamma).mean()
    return torch.tensor(res)


class PosNegCrossEntropy(nn.Module):
    def __init__(self, config, nb_model=None) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.config = config
        self.nb_model = nb_model
        self.n_tokens = (config['fdim'] + 1) * 2 + 7

    def change_batch_for_nb(self, batch): 
        no_grad_batch = {k: v.clone().detach() for k, v in batch.items()}  
        no_grad_batch['input_ids'][:, 1:5] = torch.ones(len(no_grad_batch['input_ids']), 4) * 12 #TODO this is for prompting no_grad batch
        return no_grad_batch

    def forward(self, outputs, batch):
        no_grad_batch = self.change_batch_for_nb(batch)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        labels = no_grad_batch['labels'][:, 1:].contiguous()
        y_pos = self.cross_entropy(shift_logits.view(-1, self.n_tokens), labels.view(-1))

        if self.config.train['nb_alpha']:
            with torch.no_grad():
                nb_outputs = self.nb_model(**no_grad_batch)
            if self.config.train['soft']:
                # shift_logits.softmax(2).transpose(1, 2)
                nb_labels = nb_outputs.logits.softmax(2)[:, :-1].transpose(1, 2).contiguous()
                y_neg = self.cross_entropy(shift_logits.softmax(2).transpose(1, 2), nb_labels)
            else:
                nb_labels = nb_outputs.logits.argmax(2)[:, :-1].contiguous()
                y_neg = self.cross_entropy(shift_logits.view(-1, self.n_tokens), nb_labels.view(-1))
            return y_pos - self.config.train['nb_alpha'] * y_neg
        else:
            return y_pos

def train_data_collator(batch: List[Dict[str, Any]], freegroup_dimension, tokenizer):
    def max_length_pad(batch: List, pad_token_id):
        max_length = max(map(len, batch))
        batch = map(lambda x: x + [pad_token_id] * (max_length - len(x)), batch)
        batch = torch.tensor(list(batch), dtype=int)
        return batch
    
    input_ids = list(map(lambda x: x['input_ids'], batch))
    labels = list(map(lambda x: x['labels'], batch))
    
    labels    = max_length_pad(labels[::], -100)
    labels[:, :1 + freegroup_dimension + 1 + 1] = -100
    
    input_ids = max_length_pad(input_ids, tokenizer.pad_token_id)
        
    attention_mask = torch.ones_like(input_ids)
    attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, -100)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

def gen_data_collator(batch: List[Dict[str, Any]]):
    def max_length_pad(batch: List, pad_token_id):
        max_length = max(map(len, batch))
        batch = map(lambda x: x + [pad_token_id] * (max_length - len(x)), batch)
        batch = torch.tensor(list(batch), dtype=int)
        return batch
    
    input_ids = list(map(lambda x: x['input_ids'], batch))
    input_ids = max_length_pad(input_ids, 0)
    
    return {'inputs': input_ids[:, :-1]}


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


from freegroup.tools import from_string, is_from_normal_closure, normalize

r = [1 for _ in range(3)]
def completion_ratio_R(output_words):
  result = 0
  for w in output_words['outputs']:
    if is_from_normal_closure(from_string(w), closure=r):
      result = result + 1
  return result / len(output_words['outputs'])


def multi_ratios_collate_fn(batch, tokenizer):
    batch = [el['word_str'] for el in batch]
    batch = tokenizer(batch, padding=True, return_tensors='pt', return_token_type_ids=False)
    batch['labels'] = batch['input_ids'].clone()
    batch['labels'][batch['attention_mask'] == 0] = -100
    return batch

def multi_ratios_collate_fn_inference(batch, tokenizer, tab=3):
    colon_indices = [el['word_str'].rfind(':') for el in batch]
    closures = [el['closures'] for el in batch]
    batch = [el['word_str'][:colon_indices[i] + 1] + ' '.join(el['word_str'][colon_indices[i] + 1:].split()[:tab]) for i, el in enumerate(batch)]
    batch = tokenizer(batch, padding=True, return_tensors='pt',
                      return_token_type_ids=False, padding_side='left')
    batch['input_ids'] = batch['input_ids'][:, :-1]
    batch['attention_mask'] = batch['attention_mask'][:, :-1]
    batch['closures'] = list(zip(*closures))
    for i in range(len(batch['closures'])):
        batch['closures'][i] = list(map(from_string, batch['closures'][i]))
    return batch

def train_collate_fn(batch, tokenizer):
    batch = [el['word_str'] for el in batch]
    batch = tokenizer(batch, padding=True, return_tensors='pt', return_token_type_ids=False)
    batch['labels'] = batch['input_ids'].clone()
    batch['labels'][batch['attention_mask'] == 0] = -100
    return batch

def inference_collate_fn(batch_, tokenizer):
    closures = [el['closures'] for el in batch_]
    batch = [el['word_str'] for el in batch_]
    batch = tokenizer(batch, padding=False, return_tensors='pt',
                      return_token_type_ids=False, padding_side='left')
    batch['input_ids'] = batch['input_ids'][:, :-1] # delete eof
    batch['attention_mask'] = batch['attention_mask'][:, :-1]
    batch['closures'] = list(zip(*closures))
    for i in range(len(batch['closures'])):
        batch['closures'][i] = list(map(from_string, batch['closures'][i]))
    return batch