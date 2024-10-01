from .utils import (
    from_iterator, multilabel_by_word,
    multilabel_number_to_list, multilabel_list_to_number,
)
from freegroup.tools import (
    wu_closure, normalize, flatten, batch_to_string, to_string, Mult
)
from freegroup.sampling import (
    CFGNormalClosureSampler, random_tree
)
from freegroup.sampling.helper import get_rng
from freegroup.tools.helper import remove_prefix

from copy import copy, deepcopy
from iteration_utilities import repeatfunc, unique_everseen
from itertools import islice
from functools import partial


def symmetric_commutator(
    fdim = 3,
    closures = None, samplers = None,
    length_distributions = None,
    multipliers_distribution = None,
    min_length = 1, max_length = 1000, unique = False,
    rng = 0,
    **kwargs
):
    kwargs = copy(kwargs)

    assert not (not closures is None and not samplers is None), "You should specify either `closures` or `samplers`"
    if closures is None:
        closures = [wu_closure(fdim, idx) for idx in range(fdim + 1)]

    if samplers is None:
        normal_closure_sampler_kwargs = remove_prefix('normal_closure_sampler', kwargs)
        
        samplers = [
            CFGNormalClosureSampler.build(closure = cls, fdim = fdim, **normal_closure_sampler_kwargs)
            for cls in closures
        ]

    if length_distributions is None:
        length_distributions = [lambda rng: rng.integers(6, 10) for _ in closures]
    if multipliers_distribution is None:
        multipliers_distribution = lambda rng: 1

    rng = get_rng(rng)

    random_tree_kwargs = remove_prefix('random_tree', kwargs)

    to_string_kwargs = remove_prefix('to_string', kwargs)

    def fn():
        source_words, trees = [], []
        for _ in range(multipliers_distribution(rng)):
            try:
                source_words.append([
                    sampler(dist(rng), rng = rng)
                    for sampler, dist in zip(samplers, length_distributions)
                ])
                trees.append([x[::] for x in source_words[-1]])
                rng.shuffle(trees[-1])
                trees[-1] = random_tree(trees[-1], rng = rng, **random_tree_kwargs)
            except BaseException:
                 return {'word': [], 'tree': None}
        
        tree = normalize(Mult(trees))
        word = normalize(flatten(tree))
        return {
            'source_words': source_words,
            'source_words_str': [batch_to_string(x, **to_string_kwargs) for x in source_words],
            'word': word,
            'word_str': to_string(word, **to_string_kwargs),
            'tree': tree,
            'tree_str': to_string(tree, **to_string_kwargs),
        }

    iterator = repeatfunc(fn)
    iterator = filter(lambda x: min_length <= len(x['word']), iterator)
    iterator = filter(lambda x: max_length >= len(x['word']), iterator)
    if unique: iterator = unique_everseen(iterator, key = lambda x: tuple(x['word']))

    return iterator
    

def symmetric_commutator_by_multilabel(
    multilabel = None, **kwargs,
):
    fdim = kwargs.get('fdim', 3)
    
    multilabel = multilabel_number_to_list(multilabel, fdim = fdim)

    if not 'samplers' in kwargs:
        kwargs['closures'] = [wu_closure(fdim, idx) for idx in multilabel]
    else:
        kwargs['samplers'] = [kwargs['samplers'][idx] for idx in multilabel]

    if 'length_distributions' in kwargs:
        kwargs['length_distributions'] = [kwargs['length_distributions'][idx] for idx in multilabel]
    
    iterator = symmetric_commutator(**kwargs)
    iterator = filter(lambda x: multilabel_by_word(x['word'], fdim = fdim) == multilabel, iterator)

    if kwargs.get('return_multilabel_as_number', False):
        multilabel = multilabel_list_to_number(multilabel)
    iterator = map(lambda x: {'multilabel': multilabel, **x}, iterator)
    
    return iterator
