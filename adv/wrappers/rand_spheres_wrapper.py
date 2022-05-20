'''Random Transform Wrapper for Spheres dataset (outdated)'''

import torch

from .rand_wrapper import RandWrapper


class RandSpheresWrapper(RandWrapper):
    """
    """

    def init_transforms(self, params):
        """
        Initialize some transformations that need to. Done at the begining of
        each forward pass.
        """

    def apply_transforms(self, x, params, num_total):
        """Apply all transformations in order."""

        for tf in params['transforms']:
            if tf == 'flipsign':
                mask = torch.zeros_like(x).bernoulli_(params['flipsign']['p'])
                x *= - 1 * (2 * mask - 1)
            elif tf == 'normal':
                x += torch.zeros_like(x).normal_(
                    params['normal']['mean'], params['normal']['std']).detach()
                # Use separate clipping after adding noise as other transforms
                # may require [0, 1] range
                if params['normal']['clip'] is not None:
                    x.clamp_(params['normal']['clip'][0],
                             params['normal']['clip'][1])
            elif tf == 'uniform':
                rnge = params['uniform']['range']
                x += torch.zeros_like(x).uniform_(rnge[0], rnge[1]).detach()
                if params['uniform']['clip'] is not None:
                    x.clamp_(params['normal']['clip'][0],
                             params['normal']['clip'][1])
            else:
                raise NotImplementedError(
                    'Specified transformation is not implemented.')
        return x
