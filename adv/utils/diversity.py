'''Compute diversity metric'''
import torch
import torch.nn.functional as F

from ..wrappers import RandWrapper


def compute_diversity(config, net, dataloader, x_adv=None):

    assert isinstance(net.module, RandWrapper)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diversity = {}
    num_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)
            begin = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            if begin >= config['meta']['test']['num_samples']:
                break

            targets = targets.to(device)
            if x_adv is None:
                inputs = inputs.to(device)
            else:
                if begin >= len(x_adv):
                    break
                inputs = torch.from_numpy(x_adv[begin:end]).to(device).float()
            # Get logits before applying any decision rule
            output = net(inputs, mode='test')
            # Compute diversity metric
            div = get_diversity(config['diversity'], output, targets)
            num_total += batch_size
            for key in div:
                if not key in diversity:
                    diversity[key] = 0
                diversity[key] += div[key] * inputs.size(0)

    for key in diversity:
        diversity[key] /= num_total
    return diversity


def get_diversity(params, logits, targets):
    batch_size = logits.size(0)
    num_draws = logits.size(1)
    y_pred = logits.argmax(2)
    return_dict = {}

    with torch.no_grad():
        for method in params['method']:
            if 'disagreement' in method:
                # Prediction agreement
                div = 0
                for i in range(num_draws):
                    disagree = y_pred != y_pred[:, i].view(-1, 1)
                    # "- batch_size" because we need to subtract itself for
                    # every sample in the batch
                    div += disagree.float().sum().item() - batch_size
                if method == 'mis-disagreement':
                    # Subtract disagreement caused by correct outputs
                    idx_cor = logits.argmax(-1) == targets.view(-1, 1)
                    num_cor = idx_cor.float().sum(1)
                    div -= (num_cor * (num_draws - num_cor)).sum()
                div /= batch_size * num_draws * (num_draws - 1)
            elif method == 'variance':
                prob = F.softmax(logits, dim=-1)
                mean_prob = prob.mean(1, keepdim=True)
                div = ((prob - mean_prob) ** 2).mean()
            elif method == 'avg-entropy':
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                div = - (prob * log_prob).sum(-1).mean()
            elif method == 'entropy-avg':
                prob = F.softmax(logits, dim=-1).mean(1)
                log_prob = prob.log()
                div = - (prob * log_prob).sum(-1).mean()
            elif method == 'model-uncertainty':
                prob = F.softmax(logits, dim=-1)
                avg_prob = prob.mean(1)
                log_avg_prob = avg_prob.log()
                ent_avg = - (avg_prob * log_avg_prob).sum(-1).mean()
                log_prob = F.log_softmax(logits, dim=-1)
                avg_ent = - (prob * log_prob).sum(-1).mean()
                div = ent_avg - avg_ent
            else:
                raise NotImplementedError(
                    'Specified diversity metric not implemented.')
            return_dict[method] = float(div)
    return return_dict
