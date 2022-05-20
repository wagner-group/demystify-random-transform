from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple, Type,
                    TypeVar, Union)

import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]


__all__ = ('AggMo',)


T = TypeVar('T', bound='AggMo')


class AggMo(Optimizer):
    r"""Implements Aggregated Momentum Gradient Descent.

    It has been proposed in `Aggregated Momentum: Stability Through Passive
    Damping`__

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AggMo(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

     __ https://arxiv.org/abs/1804.00325

    Note:
        Reference code: https://github.com/AtheMathmo/AggMo/blob/master/aggmo.py  # noqa
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Union[List[float], Tuple[float, ...]] = (0.0, 0.9, 0.99),
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))

        for i, beta in enumerate(betas):
            if not 0.0 <= beta < 1.0:
                msg = 'Invalid beta parameter at index 1: {}'.format(betas[i])
                raise ValueError(msg)

        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(AggMo, self).__init__(params, defaults)

    @classmethod
    def from_exp_form(
        cls: Type[T],
        params: Params,
        lr: float = 1e-3,
        a: float = 0.1,
        k: int = 3,
        weight_decay: float = 0,
    ) -> T:
        if lr <= 0.0:
            raise ValueError('Invalid parameter k: {}'.format(k))

        betas = [1 - a ** i for i in range(k)]  # type: List[float]
        return cls(params, lr, betas, weight_decay)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            total_mom = float(len(betas))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = {}
                    # param_state['step'] = {}
                    for beta in betas:
                        param_state['momentum_buffer'][
                            beta
                        ] = torch.zeros_like(
                            p.data, memory_format=torch.preserve_format
                        )
                        # param_state['step'][beta] = 1
                # update = torch.zeros_like(d_p)
                sqrt_d = np.sqrt(float(np.prod(d_p.shape[1:])))
                d_p.clamp_(-1 / sqrt_d, 1 / sqrt_d)
                for beta in betas:
                    buf = param_state['momentum_buffer'][beta]
                    buf.mul_(beta).add_(d_p)
                    # buf.mul_(beta).add_(d_p, alpha=1 - beta)
                    # print(buf[0])
                    # nu = 0.9
                    p.data.sub_(buf.sign(), alpha=group['lr'] / total_mom)
                    # p.data.sub_(buf, alpha=group['lr'] / total_mom / param_state['step'][beta])
                    # print((buf[0] / param_state['step'][beta]).abs().max())
                    # param_state['step'][beta] *= beta
                    # param_state['step'][beta] += 1
                    # update.add_(buf.sign())
                # p.data.sub_(d_p.sign(), alpha=group['lr'] * (1 - nu))
                # print(update.view(len(update), -1).abs().mean(1))
                # import pdb
                # pdb.set_trace()
                # print(param_state['step'])
        return loss
