from freegroup.tools import (
    determine_fdim, is_from_normal_closure, wu_closure
)
from freegroup.sampling import CFGNormalClosureSampler
from tqdm.auto import tqdm
from itertools import islice
from freegroup.tools.helper import remove_prefix
from copy import copy

from pickle import dump, load
from pathlib import Path

import math
import time

import re


def load_wu_samplers(fdim, **kwargs):
    return [CFGNormalClosureSampler.build(fdim = fdim, closure = wu_closure(fdim, idx)) for idx in range(fdim + 1)]


def from_iterator(iterator):
    from torch.utils.data import IterableDataset

    class Dataset(IterableDataset):
        def __init__(self): pass

        def __iter__(self): return iterator

    return Dataset()


def multilabel_list_to_number(multilabel, fdim = None):
    if isinstance(multilabel, int): return multilabel
    
    result = 0
    for flag in multilabel:
        result += 1 << flag
    return result


def multilabel_number_to_list(multilabel, fdim = None):
    if isinstance(multilabel, list): return multilabel
    
    result = []

    i = 0
    while multilabel > 0:
        if multilabel & 1: result.append(i)
        multilabel >>= 1
        i += 1
    return result
    

def multilabel_by_word(word, fdim = None):
    if word is None: return multilabel_number_to_list(0)
    
    fdim = determine_fdim(word, fdim)
    
    result = [
        idx for idx in range(fdim + 1)
        if is_from_normal_closure(word, closure = wu_closure(fdim, idx))
    ]
    result.sort()
    return result


def sample_multilabel(fdim, size = None, p = None, rng = None):
    
    rng = get_rng(rng)
    
    p = p if not p is None else {}
        
    for idx in suppressed:
        p[idx] = 0    
    proba_to_distribute = 1 - sum(p.values())
    n_undistributed = (1 << (fdim + 1)) - len(p)
    
    for idx in range(1 << (fdim + 1)):
        if idx in p: continue
        p[idx] = proba_to_distribute / n_undistributed
    
    p = [p[idx] for idx in range(1 << (fdim + 1))]
    
    return rng.choice(1 << (fdim + 1), p = p, size = size, replace = True)


def to_dataset(iterator, time_limit=None, **kwargs):
    kwargs = copy(kwargs)

    return_kwargs = remove_prefix('return', kwargs)
    tqdm_kwargs = remove_prefix('tqdm', kwargs)

    matching = return_kwargs.get('matching', '.*')
    ignoring = return_kwargs.get('ignoring', None)

    matching = re.compile(matching) if ignoring is None else None
    ignoring = None if ignoring is None else re.compile(ignoring)

    def match(key):
        if ignoring is None: return not matching.match(key) is None
        return ignoring.match(key) is None

    iterator = map(lambda x: {k: v for k, v in x.items() if match(k)}, iterator)

    size = return_kwargs.get('size', 'infinity')
    if isinstance(size, int):
        if time_limit is None:
            return list(tqdm(islice(iterator, size), total = size, **tqdm_kwargs))
        else:
            start = time.perf_counter()
            elements = []
            for i, el in tqdm(enumerate(iterator), total = size, **tqdm_kwargs):
                if (i == size) or ((time.perf_counter() - start) >= time_limit):
                    break
                elements.append(el)
            return elements



    if return_kwargs.get('dataset', True):
        return from_iterator(iterator)
        
    return iterator
