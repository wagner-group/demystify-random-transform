import math


def compute_metric(params, clean_acc, adv_acc):
    """Helper function for computing metrics.

    Args:
        params (dict): Config
        clean_acc (float): Clean accuracy
        adv_acc (float): Adversarial accuracy

    Returns:
        float: metric
    """

    if params['metric'] == 'adv_acc':
        return adv_acc
    if params['metric'] == 'clean_acc':
        return clean_acc
    if params['metric'] == 'weight_acc':
        return compute_weight_acc(params, clean_acc, adv_acc)
    if params['metric'] == 'sqrt_acc':
        return compute_sqrt_acc(params, clean_acc, adv_acc)
    raise NotImplementedError('Given metric not implemented.')


def compute_weight_acc(params, clean_acc, adv_acc):
    """Compute weighted error metric for `tune` defined as 
    clean_acc + weight * adv_acc

    Args:
        params (dict): Config
        clean_acc (float): Clean accuracy
        adv_acc (float): Adversarial accuracy

    Returns:
        float: weighted error
    """
    # Make sure that `clean_acc` above `clip_clean_acc` is useless
    if params['clip_clean_acc'] is not None:
        clean_acc = min(clean_acc, params['clip_clean_acc'])
    return clean_acc + params['adv_acc_weight'] * adv_acc


def compute_sqrt_acc(params, clean_acc, adv_acc):
    """Compute squared accuracy metric for `tune` defined as
    sqrt(clean_acc ** 2 + adv_acc ** 2)

    Args:
        params (dict): Config
        clean_acc (float): Clean accuracy
        adv_acc (float): Adversarial accuracy

    Returns:
        float: squared accuracy
    """
    # Make sure that `clean_acc` above `clip_clean_acc` is useless
    if params['clip_clean_acc'] is not None:
        clean_acc = min(clean_acc, params['clip_clean_acc'])
    return math.sqrt(clean_acc ** 2 + adv_acc ** 2)
