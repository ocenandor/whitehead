import numpy as np
from fast_bleu import SelfBLEU
from freegroup.derivatives import max_gamma_contains
from freegroup.tools import (batch_is_from_normal_closure,
                             batch_magnus_is_from_normal_closure,
                             batch_magnus_reduce_modulo_normal_closure,
                             batch_reduce_modulo_normal_closure,
                             batch_to_string, determine_fdim, generators,
                             permute_generators, reciprocal,
                             reduce_modulo_normal_closure_step,
                             substitute_generators, wu_closure)
from freegroup.tools.helper import remove_prefix
from multiprocess import Pool
from nltk.metrics.distance import *
from tqdm.auto import tqdm


def batch_few_shot_completion_ratio(outputs, fdim=3, max_shots=5, closures=None, union=False, n_proc=10):
    n = len(outputs)

    x = np.array([batch_magnus_is_from_normal_closure(outputs, closure, n_proc=n_proc) for closure in closures])
    x = x.transpose(1, 0).reshape(n // max_shots, max_shots, len(closures))
    if union:
        x = x.sum(axis=-1) / len(closures)
    else:
        x = x.prod(axis=-1)
    
    return {
        f'completion_ratio_{idx}': (x[:, :idx].mean(axis=-1)).mean()
        for idx in range(1, max_shots + 1)
        }


def batch_few_shot_reduction_ratio(outputs, fdim=3, max_shots=5, closures=None, n_proc=10):
    n = len(outputs)
    x = np.array([list(map(len, batch_magnus_reduce_modulo_normal_closure(outputs, closure, n_proc=n_proc))) for closure in closures])
    x = x.transpose(1, 0).reshape(n // max_shots, max_shots, -1)
    
    y = np.array([list(map(len, outputs)) for _ in range(len(closures))])
    y = y.transpose(1, 0).reshape(n // max_shots, max_shots, -1)
    
    x = (y - x) / (y + 1e-5)
    
    result = {}
    
    for n_shots in range(1, max_shots + 1):
        for idx in range(len(closures)):
            result.update({
                f'reductio_ratio_{n_shots}_<{idx}>': x[:, :n_shots, idx].max(axis=-1).mean()
            })
        result[f'mean_reduction_ratio_{n_shots}'] = x[:, :n_shots].mean(axis=-1).max(axis = -1).mean()
    
    return result


def batch_max_gamma_contains_ratio(outputs, fdim=3):
    gammas = max_gamma_contains(outputs, fdim=fdim)
    bincount = np.bincount(gammas)
    
    return {
        f'gamma_{idx}': bincount[idx] / len(outputs) for idx in range(1, len(bincount))
    }


def self_bleu(outputs, weights_type='exp', max_n_grams=15):
    references = list(map(lambda x: x.split(), batch_to_string(outputs)))
    
    weights = {}
    for n_grams in range(2, max_n_grams):
        if weights_type == 'exp':
            weights[n_grams] = np.array([1 << idx for idx in range(n_grams)], dtype=float)
        elif weights_type == 'lin':
            weights[n_grams] = np.array([idx + 1 for idx in range(n_grams)], dtype=float)
        else:
            raise ValueError
            
    for n_grams in range(2, max_n_grams):
        weights[n_grams] /= weights[n_grams].sum()
    
    results = {}
    scores = SelfBLEU(outputs, weights).get_score()
    for k, v in scores.items():
        results[f'self_bleu_{weights_type}_{k}'] = np.mean(v)
    return results


def dyck_path(word, closure):
    stack, path = [], [0]
    substitutions = {
        closure[0]: reciprocal(closure[1:]),
        -closure[0]: closure[1:],
    }
    for f in substitute_generators(word, substitutions):
        reduce_modulo_normal_closure_step(stack, f, closure)
        path.append(len(stack))
        
    return path

def number_of_valleys(path = None, word = None, closure = None):
    if path is None:
        path = dyck_path(word, closure)

    result = 0
    for idx in range(1, len(path) - 1):
        if path[idx - 1] > path[idx] and path[idx] < path[idx + 1] and path[idx] > 0:
            result += 1
    return result


def cycle_shift_invariant_similarity(
    s1, s2, metric_name = 'edit_distance', fdim = None, reduction = None, **metric_kwargs
):
    fdim = max(determine_fdim(s1, fdim), determine_fdim(s2, fdim))
    
    metric_fn = {
        'edit_distance': edit_distance,
        'edit_distance_align': edit_distance_align,
        'jaro_similarity': jaro_similarity,
        'jaro_winkler_similarity': jaro_winkler_similarity,
    }[metric_name]
    
    results = [
        metric_fn(s1, permute_generators(s2, fdim = fdim, shift = shift), **metric_kwargs)
        for shift in range(fdim)
    ]
    
    results = np.array(results)
    
    if reduction is None:
        return results
    
    if isinstance(reduction, str):    
        return {
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'sum': np.sum,
        }[reduction](results)
    
    return reduction(results)


def batch_cycle_shift_invariant_similarity(
    words, fdim = None, **kwargs
):
    fdim = max([determine_fdim(w, fdim) for w in words])
    
    pool_kwargs = remove_prefix('pool', kwargs)
    pool_workers = pool_kwargs.pop('workers', 1)
    
    map_kwargs = remove_prefix('map', kwargs)
    map_chunksize = map_kwargs.pop('chunksize', pool_workers * 10)
    
    tqdm_kwargs = remove_prefix('tqdm', kwargs)
    
    metric_kwargs = remove_prefix('metric', kwargs)
    metric_name = metric_kwargs.pop('name', 'edit_distance')
    metric_reduction = metric_kwargs.pop('reduction', None)
    metric_self = metric_kwargs.pop('self', 0)
    
    if kwargs: raise ValueError(f'Unknown arguments: {kwargs}')
    
    def handle(task_configuration):
        i, j = task_configuration
        return i, j, cycle_shift_invariant_similarity(
            words[i], words[j], fdim = fdim, metric_name = metric_name, reduction = metric_reduction, **metric_kwargs
        )
    
    tasks = [(i, j) for i in range(len(words)) for j in range(i + 1, len(words))]
    
    results = [[None for _ in range(len(words))] for _ in range(len(words))]
    
    with tqdm(total = len(tasks), **tqdm_kwargs) as pbar, Pool(pool_workers, **pool_kwargs) as pool:
        
        for i, j, result in pool.imap_unordered(handle, tasks, chunksize = map_chunksize, **map_kwargs):
            results[i][j] = result
            results[j][i] = result
            pbar.update(1)

    for i in range(len(words)):
        results[i][i] = [0] * fdim if metric_reduction is None else metric_self
        
    return np.array(results)
    

