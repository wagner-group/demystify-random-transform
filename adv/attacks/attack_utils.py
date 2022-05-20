from ..utils import load_dataset, is_rand_wrapper
from ..wrappers import RandWrapper
from .autoattack import AutoAttack


def setup_attack(config, net, log, mode):
    """Set up attack(s) for both adversarial training and evaluations.

    Args:
        config (dict): Main config dict
        net (torch.Module): Model to evaluate
        log (Logger): Logger
        mode (str): Mode of the attack to set up: 'train' for adversarial 
            training, or 'test' for evaluation

    Returns:
        dict: Dictionary of attacks (possible keys are 'at' and 'attack')
    """
    assert mode in ('train', 'test')
    adv = 'pgd' in config['meta']['method']

    x_train, y_train = None, None
    # if ((adv or config['meta']['train'].get('eval_with_atk', False)) and
    #         config[config['attack']['method']].get('init_mode', 1) != 1):
    #     ((x_train, y_train), _, _), _ = load_dataset(config, dataloader=False)

    attacks_dict = {}
    if adv and mode == 'train':
        # NOTE: only PGD-type attack is used for adversarial training
        log.info('Setting up attack for AT...')
        attack = _init_at(config, net, x_train, y_train)
        attack.epsilon = config['at']['epsilon']
        attacks_dict['at'] = attack

    if ((config['meta']['train'].get('eval_with_atk', False) and
         mode == 'train') or mode == 'test'):
        attacks = _init_attack(config, net, x_train, y_train, log)
        attacks_dict['attack'] = attacks

    return attacks_dict


def _init_attack(config, net, x_train, y_train, log):
    """Initialize attack modules.

    Args:
        config (dict): Main config dict
        net (torch.Module): Model to evaluate
        x_train (torch.Tensor): Training input features
        y_train (torch.Tensor): Training labels corresponding to `x_train`
        log (Logger): Logger

    Returns:
        list: List of initialized attacks
    """
    from .opt_attack import OptAttack
    from .pgd_attack import PGDAttack
    from .rand_opt_attack import RandOptAttack
    from .rand_pgd_attack import RandPGDAttack
    from .gro_attack import GROAttack

    batch_size = config['meta']['test']['batch_size']
    methods = config['attack']['method']
    if isinstance(methods, str):
        methods = [methods]
    attacks = []

    for method in methods:

        if 'pgd' in method:
            if is_rand_wrapper(net):
                attack_func = RandPGDAttack
            else:
                attack_func = PGDAttack
        elif 'opt' in method:
            if is_rand_wrapper(net):
                attack_func = RandOptAttack
            else:
                attack_func = OptAttack
        elif 'gro' in method:
            attack_func = GROAttack
        elif 'auto' in method:
            attack_func = None
        else:
            raise NotImplementedError('Specified attack not implemented!')

        # Initialize the attacks
        if 'auto' in method:
            if isinstance(net.module, RandWrapper):
                if config[method]['version'] == 'rand':
                    # Rand version of AutoAttack already uses EoT internally
                    net.module.params['attack']['rule'] = 'none'
                    net.module.params['attack']['num_draws'] = 1
                else:
                    check_rand_params(net)
            attack = AutoAttack(
                net, norm='L' + str(config['attack']['p']),
                eps=config[method]['epsilon'],
                version=config[method]['version'],
                # log_path=config['attack']['log'].handlers[0].baseFilename
            )
        else:
            # Set SGM and LinBP params
            sgm_params, linbp_params = None, None
            if config[method].get('sgm_gamma') is not None:
                sgm_params = {'gamma': config[method]['sgm_gamma'],
                              'arch': config['meta']['network']}
            if config[method].get('linbp_layer') is not None:
                linbp_params = {'start_layer': config[method]['linbp_layer'],
                                'arch': config['meta']['network']}
            if 'rand' in config['meta']['method']:
                de = config['rand']['attack'].get('double_exp', 0)
            else:
                de = None
            # Initialize attack
            attack = attack_func(net, x_train=x_train, y_train=y_train,
                                 batch_size=batch_size, log=log,
                                 p=config['attack']['p'], de=de,
                                 sgm_params=sgm_params,
                                 linbp_params=linbp_params,
                                 **config[method])
        attacks.append(attack)
    return attacks


def _init_at(config, net, x_train, y_train):
    """Initialize an attack for adversarial training."""
    from .opt_attack import OptAttack
    from .pgd_attack import PGDAttack
    from .rand_opt_attack import RandOptAttack
    from .rand_pgd_attack import RandPGDAttack

    batch_size = config['meta']['train']['batch_size']
    method = 'at'

    if is_rand_wrapper(net):
        if config[method]['method'] == 'pgd':
            attack_func = RandPGDAttack
        else:
            attack_func = RandOptAttack
    else:
        if config[method]['method'] == 'pgd':
            attack_func = PGDAttack
        else:
            attack_func = OptAttack

    sgm_params, linbp_params = None, None
    if config[method].get('sgm_gamma') is not None:
        sgm_params = {'gamma': config[method]['sgm_gamma'],
                      'arch': config['meta']['network']}
    if config[method].get('linbp_layer') is not None:
        linbp_params = {'start_layer': config[method]['linbp_layer'],
                        'arch': config['meta']['network']}

    if 'rand' in config['meta']['method']:
        de = config['rand'][method].get('double_exp', 0)
    else:
        de = None
    attack = attack_func(net, x_train=x_train, y_train=y_train, rand_mode='at',
                         batch_size=batch_size, de=de, sgm_params=sgm_params,
                         linbp_params=linbp_params, **config[method])

    return attack


def check_rand_params(net):
    """Check whether the given RandWrapper has the correct parameters for
    producing gradients for the attacks.

    Args:
        net (BaseWrapper): Model to evaluate
    """
    # if net.module.params['attack']['rule'] == 'none':
    #     assert net.module.params['attack']['num_draws'] == 1
    # else:
    #     assert net.module.params['attack']['rule'] in ['mean_probs', 'mean_logits']
    if net.params['attack']['rule'] == 'none':
        assert net.params['attack']['num_draws'] == 1
    else:
        assert net.params['attack']['rule'] in ['mean_probs', 'mean_logits']
