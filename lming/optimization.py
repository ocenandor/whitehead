import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler


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


def build_scheduler(optimizer, name, decay_duration, warmup_duration, start_lr, end_lr):
    warmup_lambda = lambda step: step / warmup_duration
    final_lambda = lambda step: end_lr / start_lr

    if name in ['Noam', 'NoamLR', 'noam']:
        return SequentialLR(optimizer,
            schedulers = [NoamLR(optimizer, warmup_duration), LambdaLR(optimizer, final_lambda)],
            milestones = [warmup_duration + decay_duration])
        
    if name in ['CosineAnnealingLR', 'cosine', 'cos']:
        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
        decay_scheduler = CosineAnnealingLR(optimizer, T_max = decay_duration, eta_min = end_lr)
        final_scheduler = LambdaLR(optimizer, final_lambda)
        return SequentialLR(optimizer,
                            schedulers = [warmup_scheduler, decay_scheduler, final_scheduler],
                            milestones = [warmup_duration, decay_duration + warmup_duration])
        
    if name in ['LinearLR', 'linear', 'lin']:
        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
        decay_scheduler = LinearLR(optimizer, start_factor = 1., end_factor = end_lr / start_lr, total_iters = decay_duration)
        final_scheduler = LambdaLR(optimizer, final_lambda)
        return SequentialLR(optimizer,
                           schedulers = [warmup_scheduler, decay_scheduler, final_scheduler],
                           milestones = [warmup_duration, decay_duration + warmup_duration])


def simulate_scheduler_and_optimizer(
    optimizer_config,
    scheduler_config,
    training_duration = 10_000,
):
    optimizer = build_optimizer((torch.nn.Parameter(torch.zeros((1, ))), ), **optimizer_config)
    scheduler = build_scheduler(optimizer, **scheduler_config)

    lrs = []
    for _ in range(training_duration):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        
    return lrs
