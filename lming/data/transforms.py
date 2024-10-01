from freegroup.sampling.helper import get_rng
from .utils import from_iterator
from freegroup.tools.helper import remove_prefix
from copy import copy


def random_choice(*iterators, p = None, rng = 0, **kwargs):
    kwargs = copy(kwargs)
    rng = get_rng(rng)

    def build():
        while True:
            idx = rng.choice(len(iterators), p = p)
            yield next(iterators[idx])

    return build()
