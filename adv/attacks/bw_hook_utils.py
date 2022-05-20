'''
This code is used with both SGM and LinBP attacks and is adapted from 
https://github.com/csdongxian/skip-connections-matter/
'''

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-6


def backward_hook(gamma, use_linbp, linbp_norm):
    # Implement SGM through grad through ReLU
    def _backward_hook_sgm(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0], )

    # Implement SGM and LinBP combined
    def _backward_hook_sgm_linbp(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_out[0], )

    # Implement SGM and LinBP combined with normalization
    def _backward_hook_sgm_linbp_norm(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            orig_norm = grad_in[0].norm() + EPS
            new_norm = grad_out[0].norm() + EPS
            return (gamma * grad_out[0] * orig_norm / new_norm, )

    if use_linbp:
        if linbp_norm:
            return _backward_hook_sgm_linbp_norm
        return _backward_hook_sgm_linbp
    return _backward_hook_sgm


def backward_hook_norm(module, grad_in, grad_out):
    # Normalize the gradient to avoid gradient explosion or vanish (SGM attack)
    std = torch.std(grad_in[0])
    if len(grad_in) == 2:
        return (grad_in[0] / std, grad_in[1] / std)
    return (grad_in[0] / std, )


def backward_hook_linbp(module, grad_in, grad_out):
    # Linear backward pass for ReLU
    return grad_out


def backward_hook_linbp_norm(module, grad_in, grad_out):
    # Linear backward pass for ReLU with gradient normalization (see Section
    # 4.2 Gradient Branching with Care in Guo et al.)
    orig_norm = grad_in[0].norm() + EPS
    new_norm = grad_out[0].norm() + EPS
    return (grad_out[0] * orig_norm / new_norm, )


def sgm_criteria(name, arch):
    # NOTE: There seems to be a very small difference in gradients between
    # splitting the gamma scaling to two ReLUs vs. applying on a single one.
    # The difference seems like a numerical instability and diverges as PGD
    # takes more steps.
    # NOTE: This is slightly different from the original code, but should
    # produce the correct effect since gradients should be scaled only on the
    # ReLUs covered by a skip connection.
    if arch in ('resnet', 'resnet50'):
        return not '0.relu' in name and 'relu2' in name
    return not '0.relu' in name and 'relu1' in name


def linbp_criteria(name, start_layer):
    # Apply to all ReLUs after start_layer
    split_name = name.split('.')
    # e.g., layerX.Y.relu
    if len(split_name) == 3:
        return (int(split_name[0][-1]) >= start_layer[0] and
                int(split_name[1]) >= start_layer[1])
    return False


def linbp_norm_criteria(name, arch, in_out_modules):
    # Normalize only ReLUs that are covered by a skip connection. Only used
    # after passing `linbp_criteria()`.
    if arch == 'resnet34':
        # For resnet34, only relu1 is covered by a skip connection
        return 'relu1' in name
    elif arch == 'resnet50':
        # For resnet50, both relu1 and relu2 are covered by a skip connection
        return ('relu1' or 'relu2') in name
    elif arch == 'resnet':
        # For pre-act resnet, only relu2 is
        return 'relu2' in name
    elif arch == 'wideresnet':
        # For wideresnet, if equalInOut, both relu1 and relu2 are. Otherwise,
        # only relu2.
        split_name = name.split('.')
        if (split_name[0] + split_name[1]) in in_out_modules:
            return ('relu1' or 'relu2') in name
        else:
            return 'relu2' in name
    else:
        raise NotImplementedError('arch not implemented!')


def register_hook_for_resnet(model, sgm_params, linbp_params):

    use_sgm = sgm_params is not None
    use_linbp = linbp_params is not None
    if not (use_sgm or use_linbp):
        return []

    # NOTE: see note in `sgm_criteria()`
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    # if arch in ['resnet50', 'resnet101', 'resnet152']:
    #     gamma = np.power(gamma, 0.5)

    if use_sgm:
        arch = sgm_params['arch']
        gamma = sgm_params['gamma']
        backward_hook_sgm = backward_hook(gamma, False, False)
        backward_hook_sgm_linbp = backward_hook(gamma, True, False)
        backward_hook_sgm_linbp_norm = backward_hook(gamma, True, True)
    if use_linbp:
        arch = linbp_params['arch']
        start_layer = linbp_params['start_layer']

    handles = []
    in_out_modules = []
    for name, module in model.named_modules():
        # Get wideresnet block that has the same number of in and out channels
        if 'wideresnet' in arch:
            split_name = name.split('.')
            if len(split_name) == 2 and 'layer' in split_name[0]:
                if module.equalInOut:
                    in_out_modules.append(name)

        if 'relu' in name:
            if (use_sgm and use_linbp and sgm_criteria(name, arch) and
                    linbp_criteria(name, start_layer)):
                # Apply hook when SGM and LinBP are both used
                if linbp_norm_criteria(name, arch, in_out_modules):
                    handles.append(module.register_full_backward_hook(
                        backward_hook_sgm_linbp_norm))
                else:
                    handles.append(module.register_full_backward_hook(
                        backward_hook_sgm_linbp))
            elif use_sgm and sgm_criteria(name, arch):
                # Apply SGM hook only
                handles.append(module.register_full_backward_hook(
                    backward_hook_sgm))
            elif use_linbp and linbp_criteria(name, start_layer):
                # Apply LinBP hook only
                if linbp_norm_criteria(name, arch, in_out_modules):
                    handles.append(module.register_full_backward_hook(
                        backward_hook_linbp_norm))
                else:
                    handles.append(module.register_full_backward_hook(
                        backward_hook_linbp))

        # Apply norm hook from SGM: e.g., 1.layer1.1, 1.layer4.2, ...
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            handles.append(module.register_full_backward_hook(backward_hook_norm))
    return handles
