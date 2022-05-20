'''Implement PGD attack for evaluating robustness of neural networks.'''
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..wrappers import RandWrapper
from .bw_hook_utils import register_hook_for_resnet
from .pgd_attack import PGDAttack


class RandPGDAttack(PGDAttack):
    """Implement PGD attack on `RandWrapper`.
    TODO: Add support for Spheres dataset.
    """

    def __init__(self, net, rand_mode='attack', maximin=None, de=0,
                 num_draws=None, rule=None, save_grad=False, **kwargs):
        # if not net.module.check_network_type(RandWrapper):
        #     raise ValueError('net must be a RandWrapper!')
        super(RandPGDAttack, self).__init__(net, **kwargs)
        if maximin is not None:
            if not isinstance(maximin, int) or maximin < 1:
                raise ValueError('maximin must be a positive integer!')
        self.rand_mode = rand_mode
        self.maximin = maximin
        self.de = de
        self.save_grad = save_grad

        self.net_module = self.net.module if hasattr(self.net, 'module') else self.net
        if rule is None:
            self.rule = self.net_module.params[rand_mode]['rule']
        else:
            self.log.info("New net's rule is specified.")
            self.rule = rule

        if num_draws is None:
            self.num_draws = self.net_module.params[rand_mode]['num_draws']
        else:
            self.log.info("New net's num_draws is specified.")
            self.num_draws = num_draws

        self.log.info(f'{self.__class__.__name__} is set up with rand_mode: '
                      f'{self.rand_mode}, rule: {self.rule}, num_draws: '
                      f'{self.num_draws}')

        # register_hook_for_resnet(self.net.module.get_orig_base_net(),
        #                          self.sgm_params, self.linbp_params)

    def _update_net_params(self):
        self.net_module.params[self.rand_mode]['rule'] = self.rule
        self.net_module.params[self.rand_mode]['num_draws'] = self.num_draws

    def attack_batch(self, x_orig, label, return_grad=False, **kwargs):

        x_orig = x_orig.to(self.device)
        label = label.to(self.device)
        if len(self.report_steps) > 0:
            x_best = []
        else:
            x_best = x_orig.clone()     # Collect best adv to return
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
        velocity, grad_vr = 0, 0
        is_fix_seed = self.net_module.params[self.rand_mode].get('fix_seed')
        # This must be the same as the default seed in RandWrapper if testing a
        # fixed-seed model. In other cases, it doesn't matter.
        seed = 1234
        use_storm_vr = self.momentum.get('vr') == 'storm'

        # Data for save_grad
        grad_list = []

        # Find starting point of attack
        x_init = self._init_adv(x_orig, label)
        # Find appropriate target labels
        target_label, num_targets = self._get_target_labels(x_orig, label)

        # Register backward hook for SGM and LinBP attack
        # TODO: adding/removing hooks too many times slow down the attack
        # significantly. In that case, the hook should be added at attack
        # initialization, but if multiple attack types are used, hook still
        # needs to be removed somehow.
        handles = register_hook_for_resnet(self.net_module.get_orig_base_net(),
                                           self.sgm_params, self.linbp_params)
        # handles = []
        clean_logits = None
        if self.loss_func == 'trades':
            with torch.no_grad():
                clean_logits = self.net(x_orig, mode=self.rand_mode, seed=seed)

        for i in range(self.num_restarts):

            # Initialize delta
            delta = self._init_delta(x_orig, x_init, batch_size, i)

            # Set label to the target class for targeted attack
            if self.targeted:
                label = target_label[:, i % num_targets]
            # Duplicate label if needed
            if self.de <= 1:
                if rule == 'eot':
                    targets = label.repeat(num_draws, 1).permute(1, 0)
                else:
                    targets = label
            else:
                targets = label.repeat(self.de, 1).permute(1, 0)
            targets = targets.reshape(-1)

            # Initialize velocity and another grad term if momentum is used
            if len(self.momentum) > 0:
                velocity = torch.zeros_like(delta)
                if 'vr' in self.momentum:
                    grad_vr = torch.zeros_like(delta)
                else:
                    grad_vr = []
                if use_storm_vr:
                    self.net_module.params[self.rand_mode]['fix_seed'] = True
                    seed = random.randint(0, 99999999)

            # ======================= Begin main PGD loop =================== #
            for step in range(self.num_steps):
                # Compute loss and gradients
                delta.requires_grad_()
                with torch.enable_grad():
                    if self.de <= 1:
                        outputs = self.net(x_orig + delta, mode=self.rand_mode, seed=seed)
                    else:
                        x_rep = x_orig.repeat(self.de, 1, 1, 1, 1).transpose(1, 0)
                        x_rep = x_rep + delta.unsqueeze(1)
                        x_rep = x_rep.reshape((self.de * batch_size,) + x_orig.size()[1:])
                        outputs = self.net(x_rep, mode=self.rand_mode, seed=seed)

                    # Compute loss and gradients
                    loss = self._compute_loss(
                        outputs, targets, gap, batch_size, num_draws, self.de,
                        clean_logits=clean_logits)
                    grad = torch.autograd.grad(loss, delta)[0].detach()

                # Check gradients for NaN
                grad = self._check_nan_grad(grad, batch_size)

                # Get seed and gradients using old delta saved for next run
                old_delta = delta.clone() if use_storm_vr else 0

                # Update perturbation
                delta, velocity, grad_vr = self._update_delta(
                    delta.detach(), x_orig, grad, batch_size,
                    velocity=velocity, grad_vr=grad_vr)

                # Collect gradients
                if return_grad:
                    grad.sign_()
                    grad_list.append(grad.view(batch_size, 1, -1).cpu())
                    if len(grad_list) == self.num_steps:
                        return grad_list
                    continue

                # Compute gradients for STORM variance reduction
                if use_storm_vr:
                    # Get a new seed for next step
                    seed = random.randint(0, 99999999)
                    old_delta.requires_grad_()
                    with torch.enable_grad():
                        outputs_vr = self.net(x_orig + old_delta,
                                              mode=self.rand_mode, seed=seed)
                        loss_vr = self._compute_loss(
                            outputs_vr, targets, gap, batch_size, num_draws, self.de)
                        grad_vr = torch.autograd.grad(
                            loss_vr, old_delta)[0].detach()

                if step + 1 in self.report_steps:
                    with torch.no_grad():
                        x_best.append(x_orig + delta.detach())

            # ======================== End main PGD loop ==================== #

            self.net_module.params[self.rand_mode]['fix_seed'] = is_fix_seed
            if self.num_restarts == 1:
                if len(self.report_steps) == 0:
                    x_best = x_orig + delta.detach()
                else:
                    x_best.append(x_orig + delta.detach())
            else:
                message = ('When multiple restarts are used, cannot report '
                           'x_adv at multiple steps.')
                assert len(self.report_steps) == 0, message
                # Compute confidence score and save best attack
                self._update_confidence(
                    x_orig + delta.detach(), label, x_best, confidence)

        # Remove the handles
        for handle in handles:
            handle.remove()

        return x_best

    def _compute_loss(self, outputs, targets, gap, batch_size, num_draws, de,
                      clean_logits=None):
        """Compute adversarial loss."""
        # This is to handle case where rule is 'eot'
        if outputs.ndim == 3:
            outputs = outputs.reshape(num_draws * batch_size, -1)

        loss = 0
        if self.ila_params is not None:
            # TODO: Use ILA loss
            pass
        elif self.loss_func == 'ce':
            # Cross entropy loss
            if self.rule == 'mean_probs':
                # 'mean_probs' rule outputs softmax prob so we need NLL loss
                loss = F.nll_loss(outputs.log(), targets, reduction='none')
            else:
                loss = F.cross_entropy(outputs, targets, reduction='none')
        elif self.loss_func in ('linear', 'hinge', 'sm-hinge', 'mat'):
            # Hinge loss (like CW attack)
            if self.loss_func == 'sm-hinge':
                assert self.rule != 'mean_probs'
                outputs = F.softmax(outputs, dim=1)
            other = self.best_other_class(outputs, targets.unsqueeze(1))
            loss = other - torch.gather(outputs, 1, targets.unsqueeze(1)).squeeze()
            if self.loss_func == 'hinge':
                if self.targeted:
                    loss = torch.max(- gap, loss)
                else:
                    loss = torch.min(gap, loss)
        elif self.loss_func == 'logits':
            # Just use logits as loss. Should be used with rule 'eot' or 'mean_logits'
            loss = - torch.gather(outputs, 1, targets.unsqueeze(1)).squeeze()
        elif self.loss_func == 'trades':
            adv_log_softmax = F.log_softmax(outputs, dim=1)
            clean_log_softmax = F.log_softmax(clean_logits, dim=1)
            loss = F.kl_div(adv_log_softmax, clean_log_softmax,
                            reduction='batchmean', log_target=True)

        if self.targeted:
            loss *= -1

        # Use maximin loss formulation focusing on the worst draws from the
        # adversary's perspective
        if self.maximin is not None:
            # NOTE: rule has to be none for maximin to work
            loss = loss.view(batch_size, num_draws).topk(
                self.maximin, dim=1, largest=False)[0].mean(1)
        elif self.rule == 'eot':
            loss /= num_draws
        elif de > 1:
            loss /= de

        return loss.sum()
