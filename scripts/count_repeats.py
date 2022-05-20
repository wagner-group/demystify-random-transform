import pickle
from os.path import join

import numpy as np

dirr = 'save/imagenette/resnet34/pgd-rand2/154/'

data = pickle.load(open(join(dirr, 'save_010422-163754.pkl'), 'rb'))
targets = pickle.load(open(join(dirr, 'save_targets.pkl'), 'rb'))['clean']['targets']

for d in ['clean', 'adv']:
    print(d)
    out = data[d]['outputs'][0].transpose(1, 0, 2)
    pred = out.argmax(-1)

    perfect_correct = ((pred == targets[:, None]).sum(1) == 1000).mean()
    print(f'perfect_correct: {perfect_correct}')

    num_majority_correct = 0
    for i, p in enumerate(pred):
        y, counts = np.unique(p, return_counts=True)
        num_majority_correct += int(y[counts.argmax()] == targets[i])
    print(f'majority_correct: {num_majority_correct / len(pred)}')
