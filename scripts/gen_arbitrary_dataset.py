import pickle
from copy import deepcopy
from itertools import islice
from os import environ

import wandb
from freegroup.sampling import freegroup
from freegroup.sampling.helper import get_rng
from freegroup.tools import to_string
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import tyro

from freegroup.sampling import CFGNormalClosureSampler

from dataclasses import dataclass


@dataclass
class Config:
    fdim: int = 2
    num_closures: int = 10
    L: int = 50
    N: int = int(1e7)
    seed: int = 0
    closure_max_len: int = 8
    max_attempts: int = 1e4


def generate_random_closure(rng, fdim, max_length=8):
    length = rng.integers(1, max_length + 1)
    closure = freegroup(fdim=fdim, length=length, rng=rng)
    return closure


def generate_closures(rng, fdim, num_closures, max_length=8):
    return [generate_random_closure(rng, fdim, max_length) for _ in range(num_closures)]


def create_samplers(closures, fdim):
    return [CFGNormalClosureSampler.build(closure=r, fdim=fdim) for r in tqdm(closures)]


def get_whitehead_multilabel(label):
    if label.startswith("r"):
        return [int(label[1:])]
    else:
        raise ValueError(f"Unknown label: {label}")


def limited_unique_generator(fn, max_attempts, L, L_increment=10):
    seen = set()
    attempt = 0
    while True:
        if attempt == max_attempts:
            L += L_increment
            attempt = 0
            print(f"max_attempts reached. Increasing L to {L}")
        else:
            item = fn(L)
            if item is not None:
                word = item["word_str"]
                if word not in seen:
                    seen.add(word)
                    attempt = 0
                    yield item
                else:
                    attempt += 1
            else:
                attempt += 1


def sample(n_samples, rng, sampler, L, label, max_attempts=1e5):
    def fn(L):
        length = rng.integers(1, L + 1)
        try:
            word = sampler(length=length, rng=rng)
            return {
                "label": label,
                "multilabel": get_whitehead_multilabel(label),
                "word_str": to_string(word),
            }
        except:
            return None

    iterator = limited_unique_generator(fn, max_attempts, L)
    return list(tqdm(islice(iterator, n_samples), total=int(n_samples)))


def main(cfg: Config):

    with open("env.json") as f:
        envs = json.load(f)

    environ.update(envs)

    environ["WANDB_DIR"] = f'/main/draft-v2/{environ["WANDB_USERNAME"]}-runs/'
    run = wandb.init(
        entity="ml-in-algebraic-topology",
        project="whitehead",
        job_type="build-dataset",
        config=vars(cfg),
    )

    rng = get_rng(cfg.seed)

    print("Generating pairs...")
    closures = generate_closures(rng, cfg.fdim, cfg.num_closures, cfg.closure_max_len)
    print(closures)

    print("Building samplers...")
    samplers = create_samplers(closures, cfg.fdim)

    dataset = []
    words_per_sampler = int(cfg.N // cfg.num_closures)

    print("Generating data...")
    for i, R in enumerate(samplers):
        dataset += sample(words_per_sampler, rng, R, cfg.L, f"r{i}", cfg.max_attempts)

    train, test = train_test_split(deepcopy(dataset), test_size=0.1)

    data_artifact = wandb.Artifact(
        f"fdim-{cfg.fdim}-arbitrary-whitehead",
        type="dataset",
        description=f"fdim={cfg.fdim}. R dataset with {cfg.num_closures} closures. Train & test. Ratio 0.9. Max word length {cfg.L}, max closure length {cfg.closure_max_len}. Total samples {cfg.N}.",
        metadata=vars(cfg),
    )

    with data_artifact.new_file("train.pkl", "wb") as file:
        pickle.dump(train, file)
    with data_artifact.new_file("test.pkl", "wb") as file:
        pickle.dump(test, file)

    run.log_artifact(data_artifact)

    wandb.finish()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
