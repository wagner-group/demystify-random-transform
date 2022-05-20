import numpy as np
import torch
import torch.optim as optim

from .bw_hook_utils import register_hook_for_resnet
from .pgd_attack import PGDAttack


class OptAttack(PGDAttack):
    """Attack based on PyTorch optimizer (only for RandWrapper)."""

    def __init__(self, net, optimizer, var_change=True,
                 normalize=False, lr_schedule=None, **kwargs):
        super(OptAttack, self).__init__(net, **kwargs)
        self.optimizer = optimizer
        self.var_change = var_change
        self.lr_schedule = lr_schedule
        self.normalize = normalize

    def attack_batch(self, x_orig, label, **kwargs):

        x_orig = x_orig.to(self.device)
        label = label.to(self.device)
        if len(self.report_steps) > 0:
            x_best = []
        else:
            x_best = x_orig.clone()
        batch_size = x_orig.size(0)
        gap = torch.tensor(self.gap, device=self.device)
        confidence = np.zeros(batch_size)
        if not self.targeted:
            confidence += 1e9

        # Find starting point of attack
        x_init = self._init_adv(x_orig, label)
        # Find appropriate target labels
        target_label, num_targets = self._get_target_labels(x_orig, label)

        # Register backward hook for SGM and LinBP attack
        handles = register_hook_for_resnet(
            self.net.module, self.sgm_params, self.linbp_params)

        # Set upper and lower bounds
        lb, ub = float('-inf'), float('inf')
        if self.clip is not None:
            if self.p == 'inf':
                lb = torch.clamp(x_orig - self.epsilon, min=self.clip[0])
                ub = torch.clamp(x_orig + self.epsilon, max=self.clip[1])
            else:
                lb, ub = self.clip[0], self.clip[1]
        z_orig = self._to_attack_space(x_orig, lb, ub)

        for i in range(self.num_restarts):

            # Add noise to the starting point if specified
            delta = self._init_delta(x_orig, x_init, batch_size, i)
            # Initialize perturbation in the transformed space
            z = self._to_attack_space(x_orig + delta, lb, ub)
            z_delta = z - z_orig

            if self.targeted:
                label = target_label[:, i % num_targets]
            targets = label.reshape(-1)

            # Set up optimizer
            if self.optimizer == 'sgd':
                opt = optim.SGD([z_delta], lr=self.step_size,
                                momentum=self.momentum.get('mu', 0),
                                nesterov=self.momentum.get('nesterov', False))
            elif self.optimizer == 'adam':
                opt = optim.Adam([z_delta], lr=self.step_size)
            elif self.optimizer == 'rmsprop':
                opt = optim.RMSprop([z_delta], lr=self.step_size)
            else:
                raise NotImplementedError('Given optimizer not implemented.')

            # Set up learning schedule
            lr_scheduler = None
            if self.lr_schedule == 'cyclic':
                lr_scheduler = optim.lr_scheduler.CyclicLR(
                    opt, self.step_size * 1e-2, self.step_size,
                    step_size_up=int(self.num_steps / 10), mode='triangular',
                    cycle_momentum=False)

            # ========================= Begin main loop ===================== #
            for step in range(self.num_steps):
                z_delta.requires_grad_()
                with torch.enable_grad():
                    opt.zero_grad()

                    # Change of variables
                    x = self._to_model_space(z_orig + z_delta, lb, ub)

                    # Compute outputs
                    outputs = self.net(x)

                    # Compute loss
                    loss = self._compute_loss(outputs, targets, gap)
                    # We have to flip sign of loss because we are minimizing here
                    loss *= -1

                    # Take optimization step
                    loss.backward()

                    # Normalize gradient before applying the update if needed
                    if self.normalize:
                        z_delta.grad = self._normalize(z_delta.grad, self.p)

                    opt.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                # TODO: handle NaN
                assert not torch.isnan(z_delta).any()

                # Project delta if change of variable is not used
                if not self.var_change:
                    # Project delta to Lp-norm ball
                    z_delta.data = self._project(
                        z_delta.data, self.p, self.epsilon, batch_size)
                    # Clip to specified domain
                    if self.clip is not None:
                        z_delta.data = (z_orig + z_delta.data).clamp(
                            self.clip[0], self.clip[1]) - z_orig

                if step + 1 in self.report_steps:
                    with torch.no_grad():
                        x = self._to_model_space(
                            z_orig + z_delta.detach(), lb, ub)
                        x_best.append(x)

            # ========================== End main loop ====================== #

            x = self._to_model_space(z_orig + z_delta.detach(), lb, ub)
            if self.num_restarts == 1:
                if len(self.report_steps) == 0:
                    x_best = x
                else:
                    x_best.append(x)
            else:
                message = ('When multiple restarts are used, cannot report '
                           'x_adv at multiple steps.')
                assert len(self.report_steps) == 0, message
                # Compute confidence score and save best attack
                self._update_confidence(x, label, x_best, confidence)

        # Remove the handles
        for handle in handles:
            handle.remove()
        return x_best

    def _to_attack_space(self, x, min_, max_):
        if not self.var_change:
            return x
        # map from [min_, max_] to [-1, +1]
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = (x - a) / b

        # from [-1, +1] to approx. (-1, +1)
        x = x * 0.99999

        # from (-1, +1) to (-inf, +inf): atanh(x)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _to_model_space(self, x, min_, max_):
        """Transforms an input from the attack space to the model space. 
        This transformation and the returned gradient are elementwise."""
        if not self.var_change:
            return x

        # from (-inf, +inf) to (-1, +1)
        x = torch.tanh(x)

        # map from (-1, +1) to (min_, max_)
        a = (min_ + max_) / 2
        b = (max_ - min_) / 2
        x = x * b + a
        return x
