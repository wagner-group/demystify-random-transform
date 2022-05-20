'''
Code adapted from https://github.com/darshank528/Project-STORM and
https://github.com/google-research/google-research/blob/e2e2237dab538ca5b96db90bca4178433274d442/storm_optimizer/storm_optimizer.py

Creating STORM optimizer class as per algorithm in the paper https://arxiv.org/abs/1905.10018
'''
import copy
import torch
from torch.optim.optimizer import Optimizer


class StormOptimizer(Optimizer):
    # Storing the parameters required in defaults dictionary
    # lr-->learning rate (NOTE: This is k in the paper)
    # c-->parameter to be swept over logarithmically spaced grid as per paper
    # w and k to be set as 0.1 as per paper
    # momentum-->dictionary storing model params as keys and their momentum term as values
    #            at each iteration(denoted by 'd' in paper)
    # gradient--> dictionary storing model params as keys and their gradients till now in a list as values
    #            (denoted by '∇f(x,ε)' in paper)
    # sqrgradnorm-->dictionary storing model params as keys and their sum of norm ofgradients till now
    #             as values(denoted by '∑G^2' in paper)

    def __init__(self, params, lr=0.1, c=100, g_max=0.1,
                 momentum={}, gradient={}, sqrgradnorm={}, max_grad={}):
        defaults = dict(lr=lr, c=c, g_max=g_max, momentum=momentum,
                        sqrgradnorm=sqrgradnorm, gradient=gradient,
                        max_grad=max_grad)
        super(StormOptimizer, self).__init__(params, defaults)

    # Returns the state of the optimizer as a dictionary containing state and param_groups as keys
    def __setstate__(self, state):
        super(StormOptimizer, self).__setstate__(state)

    # Performs a single optimization step for parameter updates
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # param_groups-->a dict containing all parameter groups
        for group in self.param_groups:
            # Retrieving from defaults dictionary
            k = group['lr']
            c = group['c']
            g_max = group['g_max']
            momentum = group['momentum']
            gradient = group['gradient']
            sqrgradnorm = group['sqrgradnorm']
            max_grad = group['max_grad']

            # Update step for each parameter present in param_groups
            for p in group['params']:
                # Calculating gradient('∇f(x,ε)' in paper)
                if p.grad is None:
                    continue
                dp = p.grad.data
                # TODO: different norm for sign grad?
                gradnorm = torch.norm(dp)

                # Updating learning rate('η' in paper)
                power = 1.0 / 3.0
                if p in sqrgradnorm:
                    scaling = (0.1 + sqrgradnorm[p]) ** power
                else:
                    scaling = (0.1 + g_max ** 3) ** power
                lr = k / scaling

                if p in max_grad:
                    max_grad[p] = max(max_grad[p], gradnorm)
                else:
                    max_grad[p] = g_max

                # Calculating and storing the momentum term(d'=∇f(x',ε')+(1-a')(d-∇f(x,ε')))
                a = min(c * lr ** 2.0, 1.0)
                if p in momentum:
                    momentum[p].sub_(gradient[p]).mul_(1 - a).add_(dp).clamp_(- max_grad[p], max_grad[p])
                else:
                    momentum[p] = dp

                print(momentum[p] - dp)

                # Updation of model parameter p
                p.data.sub_(lr * momentum[p])

                # Storing last gradient and its norm
                gradient[p] = copy.deepcopy(dp)
                sqrgradnorm[p] = gradnorm.item() ** 2

        return loss
