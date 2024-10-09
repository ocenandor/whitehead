from tokenizers import Tokenizer
from tokenizers import (
    pre_tokenizers, models, processors
)
from transformers import PreTrainedTokenizerFast

from freegroup.tools import generators


def word_level_tokenizer(
    fdim: int,
    add_commutator_tokens: bool = True, add_prompt_tokens: bool = True,
    add_post_processor: bool = True,
):
    tokenizer = Tokenizer(models.WordLevel())

    for x in generators(fdim):
        tokenizer.add_tokens([str(x)])
    
    tokenizer.add_special_tokens(['<s>', '</s>', '<pad>'])
    
    if add_commutator_tokens:
        tokenizer.add_tokens(['[', ']', ','])
    if add_prompt_tokens:
        tokenizer.add_special_tokens(['y', 'n', ':'])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
    ])

    if add_post_processor:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $ </s>",
            special_tokens=[
                ("<s>", tokenizer.token_to_id('<s>')),
                ("</s>", tokenizer.token_to_id('</s>')),
            ]
        )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token = '<s>',
        eos_token = '</s>',
        pad_token = '<pad>',
    )
    
    if add_prompt_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['y', 'n', ':']})

    return tokenizer


def build_tokenizer(name = "word-level", **kwargs):
    if name in ["word-level", "default", "word_level", "word"]:
        return word_level_tokenizer(**kwargs)
    raise ValueError('Unknown tokenizer type')
    