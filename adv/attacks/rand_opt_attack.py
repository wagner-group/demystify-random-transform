import numpy as np
import torch
import torch.optim as optim
import torch_optimizer as optim2

from .aggmo import AggMo
from .bw_hook_utils import register_hook_for_resnet
from .rand_pgd_attack import RandPGDAttack
from .sign_accsgd import SignAccSGD
from .sign_adam import SignAdam
from .sign_adamax import SignAdamax
from .storm import StormOptimizer


class RandOptAttack(RandPGDAttack):
    """Attack based on PyTorch optimizer (only for RandWrapper)."""

    def __init__(self, net, optimizer, var_change=True, normalize=False,
                 lr_schedule=None, aggmo=None, average=None, **kwargs):
        super(RandOptAttack, self).__init__(net, **kwargs)
        self.optimizer = optimizer
        self.var_change = var_change
        self.lr_schedule = lr_schedule
        self.normalize = normalize
        self.aggmo = aggmo
        self.average = average

    def _setup_optimizer(self, z_delta):
        opt_dict = {
            'rmsprop': optim.RMSprop,
            'signadam': SignAdam,
            'adamax': optim.Adamax,
            'signadamax': SignAdamax,
            'signaccsgd': SignAccSGD,
            'rprop': optim.Rprop,
            'adamod': optim2.AdaMod,
            'yogi': optim2.Yogi,
        }
        if self.optimizer in opt_dict:
            opt = opt_dict[self.optimizer]([z_delta], lr=self.step_size)
        elif self.optimizer == 'sgd':
            opt = optim.SGD([z_delta], lr=self.step_size,
                            momentum=self.momentum.get('mu', 0),
                            dampening=self.momentum.get('damp', 0),
                            nesterov=self.momentum.get('nesterov', False))
        elif self.optimizer == 'adam':
            opt = optim.Adam([z_delta], lr=self.step_size,
                             betas=(self.momentum.get('mu', 0.9), 0.999))
        elif self.optimizer == 'asgd':
            opt = optim.ASGD([z_delta], lr=self.step_size, lambd=1e-2, t0=0)
        elif self.optimizer == 'adamp':
            opt = optim2.AdamP([z_delta], lr=self.step_size, nesterov=False)
        elif self.optimizer == 'accsgd':
            opt = optim2.AccSGD([z_delta], lr=self.step_size, kappa=1000, xi=10)
        elif self.optimizer == 'adabound':
            opt = optim2.AdaBound([z_delta], lr=self.step_size, amsbound=False)
        elif self.optimizer == 'amsbound':
            opt = optim2.AdaBound([z_delta], lr=self.step_size, amsbound=True)
        elif self.optimizer == 'aggmo':
            k = max(1, self.aggmo.get('k', 3))
            if self.aggmo.get('rand_k', False):
                k = np.random.randint(1, k + 1)
            # opt = optim2.AggMo([z_delta], lr=self.step_size,
            #                    betas=[1 - (10 ** -j) for j in range(k)])
            opt = AggMo([z_delta], lr=self.step_size,
                        betas=[1 - (10 ** -j) for j in range(k)])
        elif self.optimizer == 'qhm':
            opt = optim2.QHM([z_delta], lr=self.step_size,
                             momentum=self.momentum.get('mu', 0)
                             )
        elif self.optimizer == 'qhadam':
            opt = optim2.QHAdam([z_delta], lr=self.step_size,
                                betas=(self.momentum.get('mu', 0.9), 0.999)
                                )
        elif self.optimizer == 'shampoo':
            opt = optim2.Shampoo([z_delta], lr=self.step_size, momentum=self.momentum.get('mu', 0))
        elif self.optimizer == 'storm':
            opt = StormOptimizer([z_delta], lr=self.step_size, c=100, g_max=0.1)
        else:
            raise NotImplementedError('Given optimizer not implemented.')

        return opt

    def _setup_lr_schedule(self, opt):
        lr_scheduler = None
        if self.lr_schedule == 'cyclic':
            lr_scheduler = optim.lr_scheduler.CyclicLR(
                opt, self.step_size * 1e-2, self.step_size,
                step_size_up=int(self.num_steps / 10), mode='triangular',
                cycle_momentum=False)
        elif self.lr_schedule == 'plateau':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=200, threshold=0.0001)
        elif self.lr_schedule == 'step':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                opt, [10, 20, 30, 40], gamma=0.5)
        elif self.lr_schedule == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.8)
        return lr_scheduler

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
        handles = register_hook_for_resnet(self.net_module.get_orig_base_net(),
                                           self.sgm_params, self.linbp_params)
        clean_logits = None
        if self.loss_func == 'trades':
            with torch.no_grad():
                clean_logits = self.net(x_orig, mode=self.rand_mode)

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

            if self.average:
                avg_steps = 0
                avg = torch.zeros_like(delta)

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

            # Set up optimizer and learning schedule
            opt = self._setup_optimizer(z_delta)
            lr_scheduler = self._setup_lr_schedule(opt)

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

                    # Compute loss
                    loss = self._compute_loss(outputs, targets, gap, batch_size, num_draws, self.de,
                                              clean_logits=clean_logits)
                    # We have to flip sign of loss because we are minimizing here
                    loss *= -1
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

                    # Normalize gradient before applying the update if needed
                    if self.normalize:
                        z_delta.grad = self._normalize(z_delta.grad, self.p)

                    opt.step()
                    if lr_scheduler is not None:
                        # lr_scheduler.step(loss.item())
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

                if self.average and step >= self.num_steps * 0.5:
                    avg_steps += 1
                    avg.add_(z_delta.detach())
                    print(f'avg: {step}')

            # ========================== End main loop ====================== #

            if self.average:
                x_best = self._to_model_space(z_orig + avg / avg_steps, lb, ub)
            else:
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
