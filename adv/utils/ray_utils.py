
from ray import tune
from ray.tune.stopper import Stopper

from .metric_utils import compute_metric


class CustomStopper(Stopper):
    """
    Ray Tune experiment-wide stopper that terminates when the numer of trials
    or `iterations` is at least `min_trials` and has not improved in the last
    `patience` trials.
    """

    def __init__(self, metric=None, mode="min", threshold=0., patience=5,
                 min_trials=1):
        if mode not in ('min', 'max'):
            raise ValueError(
                'The mode parameter can only be either min or max.')
        if not isinstance(patience, int) or patience < 0:
            raise ValueError('Patience must be a strictly positive integer.')
        if not isinstance(min_trials, int) or min_trials < 0:
            raise ValueError('min_trials must be a strictly positive integer.')
        self._mode = mode
        self._metric = metric
        self._patience = patience
        self._thres = threshold
        self._min_trials = min_trials
        self._trial_ids = []
        self._iterations = 0
        self.should_stop = False
        if mode == 'min':
            self._best_metric = float('inf')
        else:
            self._best_metric = - float('inf')

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        print('Total trials: ', len(self._trial_ids))
        print('Num trials after best: ', self._iterations)
        print(result[self._metric], self._best_metric)
        if trial_id in self._trial_ids:
            return self.should_stop

        if self.should_stop:
            return True

        if self.has_plateaued():
            self.should_stop = True
            return True

        self._trial_ids.append(trial_id)
        change = result[self._metric] - self._best_metric
        # If the current iteration has to stop
        if ((self._mode == 'min' and change > - self._thres) or
                (self._mode == 'max' and change < self._thres)):
            # we increment the total counter of iterations
            self._iterations += 1
        else:
            # otherwise we reset the counter
            self._iterations = 0
            self._best_metric = result[self._metric]

        self.should_stop = self.has_plateaued()
        return self.should_stop

    def has_plateaued(self):
        return (self._iterations > self._patience and
                len(self._trial_ids) >= self._min_trials)

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return self.should_stop


def trial_name_string(trial):
    """Get trial name as string.

    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    return str(trial)


def ray_update_config(config, main_config):
    """Update random transform params in `main_config` with Ray Tune `config`.

    Args:
        config (dict): Ray Tune config or evaluated search space
        main_config (dict): Main config file for random transforms

    Returns:
        dict: Updated `main_config`
    """
    for key in config:
        if key == 'point' or config[key] is None:
            # 'point' is passed automatically by dragonfly opt
            continue
        if not key in main_config['rand']['transforms']:
            raise ValueError('Search parameter is not used.')

        assert config[key] >= 0 and config[key] <= 1
        # Set `alpha` (degree of randomness) if present. Otherwise, set `p`.
        if 'alpha' in main_config['rand'][key]:
            main_config['rand'][key]['alpha'] = config[key]
        else:
            main_config['rand'][key]['p'] = config[key]
    return main_config


def ray_report(config, clean_acc, adv_acc):
    """Compute metrics and call ray report.

    Args:
        config (dict): Main config
        clean_acc (float): Clean accuracy
        adv_acc (float): Adversarial accuracy
    """
    metric = compute_metric(config['ray']['metric'], clean_acc, adv_acc)
    tune.report(clean_acc=clean_acc,
                adv_acc=adv_acc,
                **{config['ray']['metric']['metric']: metric}
                )
