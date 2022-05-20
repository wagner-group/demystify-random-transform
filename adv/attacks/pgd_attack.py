'''Implement PGD attack for evaluating robustness of neural networks.'''
import numpy as np
import torch
import torch.linalg as LA
import torch.nn.functional as F

from .base_attack import Attack
from .bw_hook_utils import register_hook_for_resnet


class PGDAttack(Attack):
    """Implement PGD attack with additional options."""

    def __init__(self, net, **kwargs):
        self._check_params(**kwargs)
        super(PGDAttack, self).__init__(net, **kwargs)

    @staticmethod
    def _check_params(p='inf', loss_func='ce', init_mode=1, ila_params=None,
                      sgm_params=None, linbp_params=None, momentum={},
                      **kwargs):
        """Verify attack parameters."""
        if p not in ['2', 'inf']:
            raise NotImplementedError('Norm not implemented (only 2 or inf)!')
        if loss_func not in ['ce', 'hinge', 'sm-hinge', 'linear', 'logits', 'trades', 'mat']:
            raise NotImplementedError('Specified loss_func is not supported!')
        if init_mode not in [1, 2, 3]:
            raise NotImplementedError('Invalid init_mode (only 1, 2, or 3)!')
        if ila_params is not None:
            if not (isinstance(ila_params, dict) and
                    isinstance(ila_params['x_base'], torch.Tensor) and
                    isinstance(ila_params['layer'], int)):
                raise ValueError('Invalid params for ILA attack!')
        if sgm_params is not None:
            if not (isinstance(sgm_params, dict) and
                    isinstance(sgm_params['arch'], str) and
                    isinstance(sgm_params['gamma'], float)):
                raise ValueError('Invalid params for SGM attack!')
        if linbp_params is not None:
            if not (isinstance(linbp_params, dict) and
                    isinstance(linbp_params['arch'], str) and
                    isinstance(linbp_params['start_layer'], (tuple, list))):
                raise ValueError('Invalid params for LinBP attack!')
        if not isinstance(momentum, dict):
            raise ValueError('Invalid value of momentum in MI-FGSM!')

    def attack_batch(self, x_orig, label, **kwargs):

        x_orig = x_orig.to(self.device)
        label = label.to(self.device)
        x_best = x_orig.clone()
        batch_size = x_orig.size(0)
        gap = torch.tensor(self.gap, device=self.device)
        confidence = np.zeros(batch_size)
        if not self.targeted:
            confidence += 1e9
        x_init = self._init_adv(x_orig, label)
        velocity = 0

        # Register backward hook for SGM attack
        handles = register_hook_for_resnet(self.net.module, self.sgm_params, self.linbp_params)

        clean_logits = None
        if self.loss_func == 'trades':
            with torch.no_grad():
                clean_logits = self.net(x_orig)

        for i in range(self.num_restarts):

            # Initialize delta
            delta = self._init_delta(x_orig, x_init, batch_size, i)

            # Initialize velocity and another grad term if momentum is used
            if len(self.momentum) > 0:
                velocity = torch.zeros_like(delta)

            for _ in range(self.num_steps):

                # Compute loss and gradients
                delta.requires_grad_()
                with torch.enable_grad():
                    logits = self.net(x_orig + delta)
                    loss = self._compute_loss(logits, label, gap, clean_logits=clean_logits)

                grad = torch.autograd.grad(loss, delta)[0].detach()
                grad = self._check_nan_grad(grad, batch_size)
                delta, velocity, _ = self._update_delta(
                    delta, x_orig, grad, batch_size, velocity=velocity)

            if self.num_restarts == 1:
                x_best = x_orig + delta.detach()
            else:
                self._update_confidence(
                    x_orig + delta.detach(), label, x_best, confidence)

        for handle in handles:
            handle.remove()

        return x_best

    # ========================== Helper functions =========================== #

    def _compute_loss(self, logits, label, gap, clean_logits=None):
        if self.loss_func in ('ce', 'mat'):
            loss = F.cross_entropy(logits, label, reduction='sum')
        elif self.loss_func in ('linear', 'hinge'):
            other = self.best_other_class(logits, label.unsqueeze(1))
            loss = other - torch.gather(
                logits, 1, label.unsqueeze(1)).squeeze()
            if self.loss_func == 'hinge':
                if self.targeted:
                    loss = torch.max(- gap, loss)
                else:
                    loss = torch.min(gap, loss)
        elif self.loss_func == 'trades':
            adv_log_softmax = F.log_softmax(logits, dim=1)
            clean_log_softmax = F.log_softmax(clean_logits, dim=1)
            loss = F.kl_div(adv_log_softmax, clean_log_softmax,
                            reduction='batchmean', log_target=True)
        else:
            raise NotImplementedError('loss not implemented!')

        if self.targeted:
            loss *= -1
        return loss.sum()

    def _check_nan_grad(self, grad, batch_size):
        """
        Check if there is any NaN gradient. If so, report and replace them with
        Gaussian noise instead of having it fail silently.
        """
        with torch.no_grad():
            # Detect zero gradient for each sample
            is_zero = grad.view(batch_size, -1).abs().sum(1) < self.EPS
            if is_zero.any():
                self.log.debug(f'{is_zero.sum()} tiny gradients detected.')

            # If there's NaN gradient, replace it with noise
            is_nan = torch.isnan(grad).view(batch_size, -1).sum(1)
            if (is_nan > 0).any():
                nan_idx = torch.nonzero(is_nan, as_tuple=True)[0]
                self.log.info(f'{len(nan_idx)} NaN gradients detected.')
                self.log.info('Ignoring and replacing with noise.')
                not_nan_idx = torch.nonzero(is_nan == 0, as_tuple=True)[0]
                # Get mean gradient norm of those that are not NaN (if any)
                if len(not_nan_idx) > 0:
                    grad_norm = LA.norm(grad[not_nan_idx].view(
                        len(not_nan_idx), -1), ord=2, dim=1).mean() + self.EPS
                else:
                    grad_norm = 1
                # Scale noise to have the same norm
                noise = torch.zeros_like(grad[nan_idx]).normal_(0, 1)
                noise = self._normalize(noise, 2) * grad_norm
                grad[nan_idx] = noise
        return grad

    def _update_confidence(self, x, label, x_best, confidence):
        """
        Update `x_best` and `confidence` in place to only keep the best
        adversarial examples so far.
        """
        batch_size = x.size(0)
        with torch.no_grad():
            y_pred = self.net(x, mode='test').cpu()
            conf = torch.gather(y_pred, 1, label.unsqueeze(1).cpu()).squeeze()
            for j in range(batch_size):
                if ((self.targeted and conf[j] > confidence[j]) or
                        (not self.targeted and conf[j] < confidence[j])):
                    x_best[j] = x[j]
                    confidence[j] = conf[j]

    def _update_delta(self, delta, x_orig, grad, batch_size, velocity=None,
                      grad_vr=0):
        """Update the perturbation `delta` for one step with given `grad`."""

        vr = self.momentum.get('vr')
        decay = self.momentum.get('decay')
        MA_WINDOW = 50      # Moving average window size

        with torch.no_grad():
            # Update velocity with momentum if specified
            if len(self.momentum) > 0:
                mu = self.momentum['mu']
                if self.momentum.get('normalize', True):
                    # Normalize gradient with l1-norm
                    grad = self._normalize(grad, 1)
                    if vr == 'storm':
                        # For STORM, we have to normalize grad again
                        grad_vr = self._normalize(grad_vr, 1)

                if vr in ('basic', 'storm'):
                    # VR is computed based on STORM and made compatible to
                    # momentum PGD attack. Only works with exponential MA
                    velocity = mu * (velocity - grad_vr) + (mu + 1) * grad
                elif decay in ('linear', 'geometric'):
                    # Decay determines type of moving average (MA)
                    if len(grad_vr) == MA_WINDOW:
                        grad_vr.pop(0)
                    grad_vr.append(grad)
                    grad_stack = torch.stack(grad_vr, dim=0)
                    weight = torch.arange(grad_stack.size(0), device=grad.device).flip(0)
                    if decay == 'linear':
                        weight = 1 - ((1 - mu) / (MA_WINDOW - 1) * weight)
                    else:
                        weight = 1 / (weight + 1)
                    grad_stack *= weight.view(-1, 1, 1, 1, 1)
                    velocity = grad_stack.sum(0)
                    grad = grad_vr
                else:
                    if decay == 'exponential':
                        velocity = mu * velocity + (1 - mu) * grad
                    else:
                        # This option is the same as original momentum PGD attack
                        velocity = mu * velocity + grad

                step = velocity.clone()
            else:
                step = grad

            # Update perturbation delta
            step = self._normalize(step, self.p)
            delta += self.step_size * step

            # Clip to epsilon ball
            delta = self._project(delta, self.p, self.epsilon, batch_size)
            # Clip to specified domain
            if self.clip is not None:
                delta = (x_orig + delta).clamp(self.clip[0], self.clip[1]) - x_orig

            # Return new delta, velocity and grad for next iter
            return delta, velocity, grad
