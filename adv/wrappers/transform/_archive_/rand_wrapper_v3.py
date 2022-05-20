from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from ..utils import set_temp_seed
from .bpi_transform import BPITransform
from .rand_wrapper import RandWrapper
from .transform import EPS, get_bpda_transform, get_transform
from .transform_picker import TransformPicker


class RandWrapperV3(RandWrapper):
    """Wrapper that provides random input transformations."""

    def __init__(self, base_net, params, input_size, device='cpu'):
        super().__init__(base_net, params, input_size, device=device)
        self.base_net = base_net
        self.params = params
        self.input_size = input_size
        self.log = params['log']
        self.transforms = {}
        self.default_mode = 'test'
        self.device = device
        self.same_on_batch = params.get('same_on_batch', False)

        # Define the aggregater network: set num_samples to max num_draws
        self.max_num_draws = params['test']['num_draws']
        self.aggregater = Aggregater(len(params['transforms']),
                                     params['num_classes'],
                                     self.max_num_draws,
                                     num_hidden=32)

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
                self.transforms[tf] = get_transform(tf, hparams,
                                                    with_params=True,
                                                    same_on_batch=self.same_on_batch)

    def apply_transforms(self, x, params, tf_order, fix_order_only=False):
        """Apply all transformations in order."""
        clip, names = params['clip'], params['transforms']
        seed = 1234 if fix_order_only else None
        t = TransformPicker(x.size(0), len(names), params.get('subset_size', len(names)),
                            tf_order=tf_order, group_size=params.get('group_size'), seed=seed)

        if tf_order != 'fixed' and self.same_on_batch:
            raise ValueError('When same_on_batch is enabled, `tf_order` must be fixed.')

        tf_params = torch.zeros((x.size(0), len(names)), device=x.device, dtype=x.dtype)

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
            x_temp, tf_param = self.transforms[tf](x[idx_x])
            # x_temp = BPITransform.apply(x[idx_x], self.transforms[tf])

            # For tf that use p only, save the boolean (1 = applied, 0 = not
            # applied). For tf that randomly select the parameter, save that
            # parameter (not necessary in range [0, 1]).
            if tf_param is None:
                tf_params[idx_x, idx_tf] = 1
            else:
                if tf_param.ndim > 1:
                    tf_param = tf_param.float().mean(tuple(range(1, tf_param.ndim)))
                assert tf_param.ndim == 1
                tf_params[idx_x, idx_tf] = tf_param

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

        return x, tf_params

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

        # NOTE: Quick hack to make sure that same_on_batch is correct
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
        ratio = 0.25
        num_ntf = int(ratio * num_draws)
        num_tf = num_draws - num_ntf
        x = inputs.repeat(num_tf, 1, 1, 1) if repeat else inputs
        if x.size(0) != batch_size * num_tf:
            raise ValueError('When `repeat` is True, `train:num_draws` must be set to `expand_input`.')

        # DEBUG
        # if params.get('save_transformed_img', False):
        #     save_image(x[:10], 'orig.png')
        #     x_orig = x.clone()

        x_tf, tf_params = self.apply_transforms(x, params, tf_order,
                                                fix_order_only=fix_order_only)

        # DEBUG
        # if params.get('save_transformed_img', False):
        #     save_image(x_tf[:10], 'transform.png')
        #     print(x_tf.min(), x_tf.max())
        #     print((x_tf - x_orig).abs().max())
        #     # raise NotImplementedError('Images are saved. Quiting.')
        #     import pdb
        #     pdb.set_trace()

        # Prepend the original untransformed input
        x_ntf = inputs.repeat(num_ntf, 1, 1, 1) if repeat else inputs
        tf_params = torch.vstack([torch.zeros_like(tf_params[:num_ntf * batch_size]), tf_params])
        x = torch.vstack([x_ntf, x_tf])
        outputs = self.base_net(x)
        outputs = self.apply_decision_rule(outputs, rule, temperature,
                                           tf_params=tf_params, num_draws=num_draws)
        return outputs

    def _apply_prob(self, idx, p):
        """Filter `idx` with passing probability `p`."""
        if p == 0:
            return []
        if p == 1:
            return idx
        selected_idx = np.random.uniform(size=len(idx)) >= 0.5
        return np.array(idx)[selected_idx]

    def apply_decision_rule(self, logits, rule, temperature, tf_params=None,
                            num_draws=None):
        """
        Apply the specified decision rule on logits (can be 2-dim or 3-dim).
        Return logits if logits has shape (N, C) or if rule is 'eot' or 
        'mean_logits'. If rule is 'majority' or 'mean_probs', return a
        softmax probability distribution over classes with shape (N, C).
        """
        batch_size = logits.size(0) // num_draws
        if rule == 'none':
            return logits
        if rule == 'eot':
            return logits.view(num_draws, batch_size, -1).transpose(0, 1)

        if rule == 'majority':
            # NOTE: majority vote does not have gradients
            logits = logits.view(num_draws, batch_size, -1).transpose(0, 1)
            y_pred = logits.argmax(2).cpu()
            num_classes = logits.size(-1)
            y_pred = np.apply_along_axis(
                lambda z, n=num_classes: np.bincount(z, minlength=n),
                axis=1, arr=y_pred) / float(y_pred.size(1))
            outputs = torch.from_numpy(y_pred).to(logits.device)
        else:
            if rule == 'mean_probs':
                softmax = F.softmax(logits / temperature, dim=-1)
                outputs = self.aggregater([tf_params, softmax], num_samples=num_draws)
            else:
                logits = self.aggregater([tf_params, logits], num_samples=num_draws)
                outputs = F.softmax(logits / temperature, dim=-1)
        outputs = outputs.clamp_min(EPS)
        return outputs


class Aggregater(nn.Module):

    def __init__(self, num_tf, num_classes, num_samples, num_hidden=64):
        super().__init__()
        self.num_samples = num_samples
        self.num_hidden = num_hidden
        self.self_attn = nn.MultiheadAttention(num_hidden, 1, batch_first=False)
        self.tf_params_head = nn.Sequential(
            nn.Linear(num_tf, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.ReLU(inplace=True),
        )
        self.outputs_head = nn.Sequential(
            nn.Linear(num_classes, num_hidden),
            nn.LayerNorm(num_hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        tf_params, softmax = x
        batch_size = softmax.size(0) // num_samples
        assert softmax.size(0) == num_samples * batch_size
        emb_params = self.tf_params_head(tf_params).view(num_samples, batch_size, -1)
        emb_softmax = self.outputs_head(softmax).view(num_samples, batch_size, -1)

        # v9: keep ratio of ntf/tf
        attn_weight = self.self_attn(emb_softmax, emb_params, emb_softmax, need_weights=True)[1]    # [bs, ns, ns]
        attn_weight = attn_weight.unsqueeze(-1)
        softmax = softmax.view(num_samples, 1, batch_size, -1).transpose(0, 2)
        out = (attn_weight * softmax).sum((1, 2)) / self.num_samples

        return out
