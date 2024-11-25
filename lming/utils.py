import wandb
import torch
from torch.optim.lr_scheduler import _LRScheduler

from typing import List, Dict, Any, Union

from pathlib import Path
from copy import copy
from freegroup.tools.helper import remove_prefix
from freegroup.tools import batch_to_string, batch_from_string
from .data.utils import multilabel_number_to_list


def prepend_yes_no_multilabel(
    words: List[str],
    multilabels: List[List[int]],
    fdim: int,
    yes_token: str = 'y',
    no_token: str = 'n',
    delimiter_token: str = ':',
):
    multilabels = [[yes_token if idx in m else no_token for idx in range(fdim + 1)] for m in multilabels]
    return [f'{" ".join(m)} {delimiter_token} {s}' for m, s in zip(multilabels, words)]


def batchify_words(
    words: Union[List[str], str, List[List[int]], List[int]],
    **kwargs,
):
    kwargs = copy(kwargs)

    if isinstance(words, list) and isinstance(words[0], int):
        words = [words]
    if isinstance(words, str):
        words = [words]

    if not isinstance(words[0], str):
        words = batch_to_string(words, **kwargs)

    return words


def batchify_multilabels(
    words: List[str],
    multilabels: Union[List[List[int]], List[int], int],
    as_number: bool = False, **kwargs,
):
    if isinstance(multilabels, int):
        multilabels = [multilabel_number_to_list(multilabels) for _ in range(len(words))]
    elif isinstance(multilabels, list) and isinstance(multilabels[0], int) and as_number:
        multilabels = [multilabel_number_to_list(m) for m in multilabels]
    return multilabels


def prepend_multilabel_prompt(
    words: Union[List[str], str, List[List[int]], List[int]],
    multilabels: Union[List[List[int]], List[int], int],
    strategy: str = 'yes-no',
    **kwargs,
):
    kwargs = copy(kwargs)

    to_string_kwargs = remove_prefix('to_string', kwargs)
    words = batchify_words(words, **to_string_kwargs)

    multilabels_kwargs = remove_prefix('multilabels', kwargs)
    multilabels = batchify_multilabels(words, multilabels, **multilabels_kwargs)
    
    assert len(multilabels) == len(words)

    strategy_kwargs = remove_prefix('strategy', kwargs)
    if strategy in ['yes-no', 'yn', 'yes_no']:
        return prepend_yes_no_multilabel(words, multilabels, **strategy_kwargs)


def to_tensor(
    words, tokenizer,
    only_input_ids=False, strip_eos_token=True,
    **kwargs
):
    kwargs = copy(kwargs)

    prompt_kwargs = remove_prefix('prompt', kwargs)
    if prompt_kwargs:
        words = prepend_multilabel_prompt(words, **prompt_kwargs)
    else:
        to_string_kwargs = remove_prefix('to_string', kwargs)
        words = batchify_words(words, **to_string_kwargs)
    kwargs['return_tensors'] = kwargs.get('return_tensors', 'pt')
    kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids', False)
    output = tokenizer(words, **kwargs)
    
    if only_input_ids:
        output = output.input_ids
        
    if only_input_ids and strip_eos_token:
        output = output[:, :-1]
        
    return output


def from_tensor(tensor, tokenizer, **kwargs):
    kwargs = copy(kwargs)
    
    from_string_kwargs = remove_prefix('from_string', kwargs)
    
    kwargs['skip_special_tokens'] = False
    strings = tokenizer.batch_decode(tensor, **kwargs)
    strings = [el.split(':')[-1].split('</s>')[0] for el in strings]
    return batch_from_string(strings, **from_string_kwargs)


def checkpoints_path(sync = False):
    assert not wandb.run is None, "wandb isn't initialized"

    if sync: Path(wandb.run.dir, "checkpoints")
    return Path(wandb.run.dir, "..", "checkpoints")


def download_artifact(artifact_name):
    from os import environ

    assert not wandb.run is None, "wandb isn't initialized"

    download_path = environ.get('WANDB_DIR', None)
    if not download_path is None: download_path = f'{download_path}/{artifact_name}'

    return wandb.run.use_artifact(artifact_name).download(root=download_path)


def build_optimizer(parameters, name, **kwargs):
    return getattr(torch.optim, name)(parameters, **kwargs)


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_duration, **kwargs):
        self.warmup_duration = warmup_duration
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_duration ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_duration ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]    


def build_scheduler(optimizer, name, **kwargs):
    if name in ['Noam', 'NoamLR', 'noam']:
        return NoamLR(optimizer, **kwargs)

    warmup_duration = kwargs.pop('warmup_duration')
    warmup_start_factor = kwargs.pop('warmup_start_factor')
    warmup_end_factor = kwargs.pop('warmup_end_factor')
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
        start_factor = warmup_start_factor, end_factor = warmup_end_factor, total_iters = warmup_duration
    )
    
    base_scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)
    
    return torch.optim.lr_scheduler.SequentialLR(optimizer, 
                schedulers = [warmup_scheduler, base_scheduler], milestones = [warmup_duration])

