import numpy as np
import torch
import torch.optim as optim
import torch_optimizer as optim2

from .bw_hook_utils import register_hook_for_resnet
from .rand_pgd_attack import RandPGDAttack
from .sign_accsgd import SignAccSGD
from .sign_adam import SignAdam
from .sign_adamax import SignAdamax
from .sign_radam import SignRAdam


class GROAttack(RandPGDAttack):
    """Attack based on PyTorch optimizer (only for RandWrapper)."""

    def __init__(self, net, optimizer, var_change=True,
                 normalize=False, lr_schedule=None, **kwargs):
        super(GROAttack, self).__init__(net, **kwargs)
        self.optimizer = optimizer
        self.var_change = var_change
        self.lr_schedule = lr_schedule
        self.normalize = normalize

    def _setup_optimizer(self, z_delta):
        opt_dict = {
            'rmsprop': optim.RMSprop,
            'adam': optim.Adam,
            'signadam': SignAdam,
            'adamax': optim.Adamax,
            'signadamax': SignAdamax,
            'radam': optim.RAdam,
            'signradam': SignRAdam,
            'nadam': optim.NAdam,
            'signaccsgd': SignAccSGD,
            'aggmo': optim2.AggMo,
            'rprop': optim.Rprop,
        }
        if self.optimizer in opt_dict:
            opt = opt_dict[self.optimizer]([z_delta], lr=self.step_size)
        elif self.optimizer == 'sgd':
            opt = optim.SGD([z_delta], lr=self.step_size,
                            momentum=self.momentum.get('mu', 0),
                            dampening=self.momentum.get('damp', 0),
                            nesterov=self.momentum.get('nesterov', False))
        elif self.optimizer == 'asgd':
            opt = optim.ASGD([z_delta], lr=self.step_size, lambd=1e-2, t0=0)
        elif self.optimizer == 'adamp':
            opt = optim2.AdamP([z_delta], lr=self.step_size, nesterov=False)
        elif self.optimizer == 'accsgd':
            opt = optim2.AccSGD([z_delta], lr=self.step_size, kappa=1000, xi=10)
        else:
            raise NotImplementedError('Given optimizer not implemented.')

        return opt

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

        # Set up parameters specific to RandWrapper
        self._update_net_params()
        rule = self.rule
        num_draws = self.num_draws
        assert self.de <= 1 or rule in ['mean_probs', 'mean_logits']

        # Find starting point of attack
        x_init = self._init_adv(x_orig, label)
        # Find appropriate target labels
        target_label, num_targets = self._get_target_labels(x_orig, label)

        # Register backward hook for SGM and LinBP attack
        handles = register_hook_for_resnet(self.net.module.get_orig_base_net(),
                                           self.sgm_params, self.linbp_params)

        grads = []

        # Set upper and lower bounds
        lb, ub = float('-inf'), float('inf')
        if self.clip is not None:
            if self.p == 'inf':
                lb = torch.clamp(x_orig - self.epsilon, min=self.clip[0])
                ub = torch.clamp(x_orig + self.epsilon, max=self.clip[1])
            else:
                lb, ub = self.clip[0], self.clip[1]
        z_orig = self._to_attack_space(x_orig, lb, ub)

        if self.de > 1:
            z_rep = z_orig.repeat(self.de, 1, 1, 1, 1).transpose(1, 0)
            lb_rep = lb.repeat(self.de, 1, 1, 1, 1).transpose(1, 0)
            ub_rep = ub.repeat(self.de, 1, 1, 1, 1).transpose(1, 0)

        for i in range(self.num_restarts):

            # Add noise to the starting point if specified
            delta = self._init_delta(x_orig, x_init, batch_size, i)
            # Initialize perturbation in the transformed space
            z = self._to_attack_space(x_orig + delta, lb, ub)
            z_delta = z - z_orig

            if self.targeted:
                label = target_label[:, i % num_targets]
            # Duplicate label if needed
            if self.de <= 1:
                targets = label
                if rule == 'eot':
                    targets = targets.repeat(num_draws, 1).permute(1, 0)
            else:
                targets = label.repeat(self.de, 1).permute(1, 0)
            targets = targets.reshape(-1)

            # Set up optimizer
            opt = self._setup_optimizer(z_delta)

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
                    if self.de <= 1:
                        x = self._to_model_space(z_orig + z_delta, lb, ub)
                    else:
                        z = z_rep + z_delta.unsqueeze(1)
                        x = self._to_model_space(z, lb_rep, ub_rep).reshape(
                            (self.de * batch_size,) + x_orig.size()[1:])

                    # Compute outputs
                    outputs = self.net(x, mode=self.rand_mode)
                    nt_outputs = self.net(x.clone(), rand=False)
                    nt_loss = self._compute_loss(nt_outputs, targets, gap, batch_size, 1, 1)
                    # other = self.best_other_class(nt_outputs, targets.unsqueeze(1))
                    # nt_loss = other - torch.gather(nt_outputs, 1, targets.unsqueeze(1)).squeeze()
                    # nt_loss = torch.min(1e-3, nt_loss).mean()
                    # nt_loss = nt_loss.clamp_max_(1e-3)
                    # nt_loss = nt_loss.mean()

                    # Compute loss
                    loss = self._compute_loss(outputs, targets, gap, batch_size, num_draws, self.de)
                    # We have to flip sign of loss because we are minimizing here
                    alpha = 0.5
                    loss = -1 * (alpha * loss + (1 - alpha) * nt_loss)
                    loss.backward()

                    # DEBUG
                    # grads.append(z_delta.grad.clone())
                    # if step == 20:
                    #     import pdb
                    #     pdb.set_trace()
                    #     grads = torch.stack(grads)
                    #     std, mean = torch.std_mean(grads[:, 0], 0, True)
                    #     torch.histogram(mean.cpu())
                    #     torch.quantile(mean, 0.95)
                    # print(loss.item())

                    # q = 0.1
                    # lo = torch.quantile(z_delta.grad.view(batch_size, -1), q, 1, keepdim=True)
                    # hi = torch.quantile(z_delta.grad.view(batch_size, -1), 1 - q, 1, keepdim=True)
                    # z_delta.grad.clamp_(lo[:, :, None, None], hi[:, :, None, None])

                    # Normalize gradient before applying the update if needed
                    if self.normalize:
                        z_delta.grad = self._normalize(z_delta.grad, self.p)

                    opt.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                # TODO: handle NaN
                if torch.isnan(z_delta).any():
                    raise ValueError('NaN update found. Quiting...')

                # Project delta if change of variable is not used
                if not self.var_change:
                    # Project delta to Lp-norm ball
                    z_delta.data = self._project(z_delta.data, self.p, self.epsilon, batch_size)
                    # Clip to specified domain
                    if self.clip is not None:
                        z_delta.data = (z_orig + z_delta.data).clamp(
                            self.clip[0], self.clip[1]) - z_orig

                if step + 1 in self.report_steps:
                    with torch.no_grad():
                        x = self._to_model_space(z_orig + z_delta.detach(), lb, ub)
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
