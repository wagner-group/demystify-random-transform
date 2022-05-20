from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from ..utils import set_temp_seed
from .transform import (EPS, get_bpda_transform, get_transform)
from .transform_picker import TransformPicker
from .wrapper import Wrapper
from .bpi_transform import BPITransform


class RandWrapper(Wrapper):
    """Wrapper that provides random input transformations."""

    def __init__(self, base_net, params, input_size, device='cpu'):
        super().__init__()
        self.base_net = base_net
        self.params = params
        self.input_size = input_size
        self.log = params['log']
        self.transforms = {}
        self.default_mode = 'test'
        self.device = device
        self.same_on_batch = params.get('same_on_batch', False)

        # Initialize transformations
        self._get_transform_func()

    def _get_transform_func(self):
        for tf in self.params['transforms']:
            hparams = deepcopy(self.params[tf])
            hparams['input_size'] = self.input_size
            if 'bpda' in tf:
                self.transforms[tf] = get_bpda_transform(
                    hparams, self.device, same_on_batch=self.same_on_batch)
            else:
                self.transforms[tf] = get_transform(
                    tf, hparams, with_params=False, same_on_batch=self.same_on_batch)

    def apply_transforms(self, x, params, tf_order, fix_order_only=False):
        """Apply all transformations in order."""
        clip, names = params['clip'], params['transforms']
        seed = 1234 if fix_order_only else None
        t = TransformPicker(
            x.size(0), len(names), params.get('subset_size', len(names)),
            tf_order=tf_order, group_size=params.get('group_size'), seed=seed)

        if tf_order != 'fixed' and self.same_on_batch:
            raise ValueError('When same_on_batch is enabled, tf_order must be fixed.')

        # Loop through TransformPicker to apply transforms in the given order
        for idx_tf, idx_x in t:
            tf = names[idx_tf]
            # Apply the transform in-place with probability p
            p = params.get('set_all_p', None) or params[tf].get('p', 1.)
            if self.same_on_batch:
                idx_x = np.arange(x.size(0)) if np.random.rand() >= 0.5 else []
            else:
                idx_x = self._apply_prob(idx_x, p)
            if len(idx_x) == 0:
                continue
            # Apply a transform on a subset of inputs
            x_temp = self.transforms[tf](x[idx_x])
            # x_temp = BPITransform.apply(x[idx_x], self.transforms[tf])

            # Check for NaN output
            is_nan = torch.isnan(x_temp).reshape(len(idx_x), -1).sum(1)
            if (is_nan > 0).any():
                with torch.no_grad():
                    # If there's NaN, use original input
                    nan_idx = torch.nonzero(is_nan, as_tuple=True)[0]
                    self.log.info(f'{len(nan_idx)} NaN output(s) detected on '
                                  f'{tf}. Reverting the transform.')
                    x_temp[nan_idx] = x[idx_x][nan_idx]

            if clip is not None:
                x_temp = x_temp.clamp(clip[0], clip[1])
            x[idx_x] = x_temp

        return x

    def forward(self, inputs, rand=True, params=None, mode=None, seed=1234, **kwargs):
        """
        Run a forward pass which calls on self._forward. This is a wrapper for
        handling the random seed context and mode-specific parameters.
        """
        if not rand:
            return self.base_net(inputs)
        if params is None:
            params = self.params
        if mode is None:
            mode = self.default_mode

        # TODO: Quick hack to make sure that same_on_batch is correct
        seed = 1234 if self.same_on_batch else seed

        if not params[mode].get('fix_seed', False):
            return self._forward(inputs, params=params, **params[mode], **kwargs)
        # Context to set random seed temporarily
        with set_temp_seed(seed):
            output = self._forward(inputs, params=params, **params[mode], **kwargs)
        return output

    def _forward(self, inputs, num_draws=None, params=None, rule=None,
                 temperature=1., tf_order='random', repeat=True,
                 fix_order_only=False, **kwargs):
        """Helper function called by forward.

        Args:
            inputs (torch.Tensor): Input images.
            num_draws (int, optional): Number of Monte Carlo samples (n).
            params (dict, optional): Parameters for the transformations. 
            rule (str, optional): Decision rule (options: 'eot', 'majority',
                'mean_probs', 'mean_logits')
            temperature (float, optional): Temperature scaling for softmax. 
                Defaults to 1.
            tf_order (str, optional): Permutation method of the transforms. 
                See TransformPicker for details. Defaults to 'random'.
            repeat (bool, optional): Whether to make copies of the inputs. 
                Defaults to True.
            fix_order_only (bool, optional): Whether to fix the transform
                permutation. Defaults to False.

        Returns:
            torch.Tensor: Output logits or softmax probability. Exact shape
                also depends on `rule`. If `rule` is 'eot', the shape is 
                (batch_size, num_draws, num_classes). Otherwise, the shape is
                (batch_size, num_classes).
        """
        batch_size = inputs.size(0)
        x = inputs.repeat_interleave(num_draws, 0) if repeat else inputs
        if x.size(0) != batch_size * num_draws:
            raise ValueError('When `repeat` is True, `train:num_draws` must '
                             'be set to `expand_input`.')

        if params.get('save_transformed_img', False):
            save_image(x[:10], 'orig.png')
            x_orig = x.clone()

        x_tf = self.apply_transforms(x, params, tf_order, fix_order_only=fix_order_only)

        if params.get('save_transformed_img', False):
            save_image(x_tf[:10], 'transform.png')
            print(x_tf.min(), x_tf.max())
            print((x_tf - x_orig).abs().max())
            # raise NotImplementedError('Images are saved. Quiting.')
            import pdb
            pdb.set_trace()

        outputs = self.base_net(x_tf)
        if num_draws > 1:
            outputs = outputs.view(batch_size, num_draws, -1)
        outputs = self.apply_decision_rule(outputs, rule, temperature)
        return outputs

    def _apply_prob(self, idx, p):
        """Filter `idx` with passing probability `p`."""
        if p == 0:
            return []
        if p == 1:
            return idx
        return np.array(idx)[np.random.uniform(size=len(idx)) >= 0.5]

    @staticmethod
    def apply_decision_rule(logits, rule, temperature):
        """
        Apply the specified decision rule on logits (can be 2-dim or 3-dim).
        Return logits if logits has shape (N, C) or if rule is 'eot' or 
        'mean_logits'. If rule is 'majority' or 'mean_probs', return a
        softmax probability distribution over classes with shape (N, C).
        """
        if logits.dim() == 2:
            return logits
        if rule == 'eot':
            logits = logits.squeeze(1)
            return logits

        if rule == 'majority':
            # NOTE: majority vote does not have gradients
            y_pred = logits.argmax(2).cpu()
            num_classes = logits.size(-1)
            y_pred = np.apply_along_axis(
                lambda z, n=num_classes: np.bincount(z, minlength=n),
                axis=1, arr=y_pred) / float(y_pred.size(1))
            outputs = torch.from_numpy(y_pred).to(logits.device)
            outputs.clamp_min_(EPS)
        elif rule == 'mean_logits':
            outputs = logits.mean(1) / temperature
        elif rule == 'mean_probs':
            outputs = F.softmax(logits / temperature, dim=-1).mean(1)
            outputs.clamp_min_(EPS)
        else:
            raise NotImplementedError('Given rule is not implemented!')

        return outputs
