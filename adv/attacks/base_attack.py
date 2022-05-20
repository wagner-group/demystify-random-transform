import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import linalg as LA

from ..utils import set_temp_seed
from ..wrappers import Wrapper


class Attack(ABC):
    EPS = 1e-9

    def __init__(self, net, x_train=None, y_train=None, batch_size=200,
                 log=None, p='inf', targeted=False, epsilon=None,
                 step_size=None, num_steps=None, num_restarts=1,
                 loss_func='ce', gap=0., init_mode=1, random_start=True,
                 clip=None, ila_params=None, sgm_params=None, linbp_params=None,
                 momentum={}, report_steps=[], **kwargs):

        assert (epsilon and step_size and num_steps) is not None, \
            'epsilon, step_size, and num_steps must be specified'
        self.net = net
        self.x_train = x_train
        self.y_train = y_train
        if log is None:
            # Create dummy logger
            self.log = logging.getLogger('dummy')
        else:
            if not isinstance(log, logging.Logger):
                raise TypeError('log must be a Logger.')
            else:
                self.log = log
        self.device = next(net.parameters()).device
        self.p = p
        self.targeted = targeted
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_restarts = num_restarts
        self.loss_func = loss_func
        self.gap = gap
        self.init_mode = init_mode
        self.random_start = random_start
        self.clip = clip
        self.ila_params = ila_params
        self.sgm_params = sgm_params
        self.linbp_params = linbp_params
        self.momentum = momentum
        self.report_steps = report_steps

        if x_train is not None and y_train is not None:
            self.log.debug('Getting predictions for clean training data...')
            num_train = x_train.size(0)
            with torch.no_grad():
                y_pred = []
                num_batches = np.ceil(num_train / batch_size).astype(np.int32)
                for i in range(num_batches):
                    begin = i * batch_size
                    end = (i + 1) * batch_size
                    y_pred.append(net(x_train[begin:end].to('cuda'),
                                      mode='test').argmax(1).cpu())
                self.y_pred = torch.cat(y_pred, dim=0).long()
            assert self.y_pred.ndim == 1
        self.log.debug('Attack setup done.')

    @abstractmethod
    def _check_params(self, **kwargs):
        pass

    @abstractmethod
    def attack_batch(self, x_orig, label, **kwargs):
        pass

    def __call__(self, x_orig, label, batch_size=None, **kwargs):

        batch_size = x_orig.size(0) if batch_size is None else batch_size
        x_adv = torch.zeros_like(x_orig)
        num_batches = int(np.ceil(x_orig.size(0) / batch_size))

        self.log.debug('Starting attack...')
        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            x_adv[begin:end] = self.attack_batch(
                x_orig[begin:end], label[begin:end], **kwargs)
            self.log.debug(f'Batch {i} finished.')
        return x_adv

    def _init_delta(self, x_orig, x_init, batch_size, restart):
        """Initialize samples for each random restart."""
        with torch.no_grad():
            # For init_mode = 3, set new starting point at every restart
            if x_orig.ndim != x_init.ndim:
                x = self._project_eps(x_orig, x_init[restart], batch_size)
            else:
                x = self._project_eps(x_orig, x_init, batch_size)
            delta = torch.zeros_like(x_orig)
            delta += x - x_orig

            # Add noise to the starting point if specified
            if self.random_start:
                noise = torch.zeros_like(x_orig)
                # Make sure that each random restart uses a different seed.
                # This line is needed because seed is also set in RandWrapper.
                with set_temp_seed(restart * 1234):
                    if self.p == '2':
                        noise.normal_(0, self.epsilon)
                        noise = self._project(
                            noise, self.p, self.epsilon, batch_size)
                    elif self.p == 'inf':
                        noise.uniform_(- self.epsilon, self.epsilon)
                delta += noise

            # Clip x + noise to make sure it stays in the domain
            if self.clip is not None:
                delta = (x_orig + delta).clamp(self.clip[0],
                                               self.clip[1]) - x_orig

        return delta

    @staticmethod
    def best_other_class(logits, exclude):
        """Returns the index of the largest logit, ignoring the class that
        is passed as `exclude`."""
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, exclude, 1)
        # make logits that we want to exclude a large negative number
        other_logits = logits - y_onehot * 1e9
        return other_logits.max(1)[0]

    def find_neighbor_diff_class(self, x, label):
        """Find the nearest training sample to x that has a different label"""

        nn = torch.zeros((x.size(0), ), dtype=torch.long)
        norm = 2 if self.p == '2' else np.inf

        for i in range(x.size(0)):
            dist = (x[i].cpu() - self.x_train).view(
                self.x_train.size(0), -1).norm(norm, 1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i].cpu())[0]
            dist[ind] += 1e9
            nn[i] = dist.argmin()
        return self.x_train[nn].to(x.device)

    def find_kth_neighbor_diff_class(self, x, label):
        """Find k-th nearest training sample to x that has a different label"""

        x_topk = torch.zeros((self.num_restarts, ) + x.size())
        norm = 2 if self.p == '2' else np.inf

        for i in range(x.size(0)):
            dist = (x[i].cpu() - self.x_train).view(
                self.x_train.size(0), -1).norm(norm, 1)
            # we want to exclude samples that are classified to the
            # same label as x_orig
            ind = np.where(self.y_pred == label[i].cpu())[0]
            dist[ind] += 1e9
            topk = torch.topk(dist, self.num_restarts, largest=False)[1]
            x_topk[:, i] = self.x_train[topk]

        return x_topk.to(x.device)

    def _get_target_labels(self, x_orig, label):
        """Compute target classes when `targeted` is True."""
        batch_size = x_orig.size(0)
        if self.targeted:
            with torch.no_grad():
                y_pred = self.net(x_orig, mode='test')
                # Use one target per restart
                num_targets = min(self.num_restarts, y_pred.size(-1))
                target_label = torch.zeros((batch_size, num_targets),
                                           device=self.device, dtype=torch.long)
                y_pred[torch.arange(batch_size), label] -= 1e9
                target_label = torch.argsort(y_pred, descending=True)
            return target_label[:, :num_targets], num_targets
        return None, None

    def _init_adv(self, x_orig, label):
        """Find initialized samples before starting perturbing with PGD.

        Args:
            x_orig (torch.tensor): Given test inputs
            label (torch.tensor): Labels corresponding to `x_orig`
            p (str): Lp-norm of the perturbation
            epsilon (float): Maximum norm of perturbation
            num_restarts (int): Number of random restarts to use
            init_mode (int): Initialization mode for adversarial examples

        Raises:
            ValueError: Invalida `init_mode`. Must be 1, 2, or 3.

        Returns:
            torch.tensor: Initialized samples
        """
        # Initialize starting point of PGD according to specified init_mode
        x_init = None
        if self.init_mode == 1:
            # init w/ original point
            x_init = x_orig
        elif self.init_mode == 2:
            # init w/ nearest training sample that has a different label
            x_top1 = self.find_neighbor_diff_class(x_orig, label)
            x_init = self._project_eps(x_orig, x_top1, x_orig.size(0))
        elif self.init_mode == 3:
            # init w/ k nearest training samples that have a different label
            x_init = self.find_kth_neighbor_diff_class(x_orig, label)
        return x_init

    @staticmethod
    def _project(x, p, epsilon, batch_size):
        """Project `x` to an Lp-norm ball with radius `epsilon`."""
        if p == '2':
            x_norm = x.view(batch_size, -1).renorm(2, 0, epsilon)
            x_norm = x_norm.view(x.size())
        elif p == 'inf':
            x_norm = x.clamp(- epsilon, epsilon)
        else:
            x_norm = x
        return x_norm

    def _project_eps(self, x_orig, x_nn, batch_size):
        """Project `x_nn` onto an epsilon-ball around `x_orig`. The ball is
        specified by `p` and `epsilon`."""
        delta = self._project(x_nn - x_orig, self.p, self.epsilon, batch_size)
        return x_orig + delta

    def _normalize(self, x, p):
        """Normalize `x` to lp-norm of 1. Assume first dim is batch."""
        if p == 'inf':
            return torch.sign(x)
        norm = LA.norm(x.view(x.size(0), -1), ord=int(p), dim=1) + self.EPS
        for _ in range(x.ndim - 1):
            norm = norm.unsqueeze(-1)
        return x / norm
