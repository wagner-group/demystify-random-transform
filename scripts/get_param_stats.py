import pdb
from adv.wrappers.rand_wrapper import TRANSFORMS
import os
import pickle
import numpy as np
import pprint

DIR = './save/cifar10/resnet/rand/'


settings = [f for f in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, f))]

stats = {}

for setting in settings:
    cur_dir = os.path.join(DIR, setting)
    trans = setting.split('-')
    num_trans = len(trans)
    exps = [os.path.join(cur_dir, e) for e in os.listdir(cur_dir) if
            os.path.isdir(os.path.join(cur_dir, e))]

    if not num_trans in stats:
        stats[num_trans] = {}
        for tf in TRANSFORMS:
            stats[num_trans][tf] = []

    for e in exps:
        if os.path.isfile(e + '/rand.cfg'):
            config = pickle.load(open(e + '/rand.cfg', 'rb'))
        else:
            continue

        for key in config.keys():
            if key in config['transforms']:
                params = config[key].keys()
                if 'std' in params:
                    param = config[key]['std']
                elif 'range' in params:
                    param = config[key]['range'][1]
                elif 'alpha' in params:
                    param = config[key]['alpha']
                else:
                    param = config[key]['p']
                if param > 0:
                    stats[num_trans][key].append(param)

for key in stats:
    print('\n' + key)
    for tf in stats[key]:
        if tf in ['edsr', 'scale', 'rotate', 'translate', 'equalize', 'shear']:
            continue
        if len(stats[key][tf]) >= 5:
            mean = np.mean(stats[key][tf])
            std = np.std(stats[key][tf])
            # stats[key][tf] = {'mean': np.mean(stats[key][tf]),
            #                   'std': np.std(stats[key][tf])}
            stats[key][tf] = [max(0.001, mean - std), min(1, mean + std)]
        else:
            try:
                params = config[tf].keys()
                if 'std' in params:
                    stats[key][tf] = [0.01, 0.3]
                elif 'range' in params:
                    stats[key][tf] = [0.01, 0.3]
                elif 'alpha' in params:
                    stats[key][tf] = [0.01, 0.3]
                else:
                    stats[key][tf] = [0., 0.5]
            except:
                pass
        print(f'{tf}: {stats[key][tf]}')


# pprint.pprint(stats)
# pdb.set_trace()
