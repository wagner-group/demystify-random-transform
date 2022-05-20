import argparse

import numpy as np
from ray.tune import ExperimentAnalysis


def main():
    METRIC = 'weight_acc'
    # RAND = args.transforms
    # EXP = args.exp
    # DEPTH = args.depth
    # if args.dataset == 'cifar':
    #     analysis = Analysis(f'./save/cifar10/resnet/rand/{RAND}/{EXP}/tune/')
    # else:
    #     analysis = Analysis(f'./save/imagenette/resnet34/rand/{RAND}/{EXP}/tune/')

    # TODO
    # analysis = ExperimentAnalysis('./save/cifar10/resnet/pgd-rand2/33/tune/')
    # analysis = ExperimentAnalysis('./save/imagenette/resnet34/pgd-rand2/215/tune/')
    analysis = ExperimentAnalysis(
        './save/imagenette/resnet34/pgd-rand/affine-boxblur-crop-erase-fft-gaussblur-gray-gray1-graymix-hflip-motionblur-normal-pepper-poisson-salt-sharp-solarize-speckle-swirl-uniform-vflip/236/tune/')

    df = analysis.dataframe()

    idx = df[METRIC].argmax()
    # idx = np.array(df[METRIC].argsort())[-3]

    # Get trial_id from the max idx
    trial_id = df['trial_id'][idx]
    # Sorted trials by their timestamp
    sorted_df = df.sort_values('timestamp')
    # Get the max trial's order by querying with the id
    trial_order = sorted_df['trial_id'].tolist().index(trial_id)
    # print(f'{RAND}_exp{EXP}')
    print(f'Trial {trial_order}: {METRIC}={df[METRIC][idx]:.2f}, clean_acc='
          f'{df["clean_acc"][idx]}, adv_acc={df["adv_acc"][idx]}.')
    # Get date
    print(f'Date of best trial: {df["date"][idx]}')

    for key in df.keys():
        if 'config/' in key and 'point' not in key:
            print(f'{key}={df[key][idx]:.4f}')

    # max
    print(sorted(df[METRIC])[-10:])
    print(np.std(sorted(df[METRIC])[-10:]))

    print(f'Number of trials run: {len(df)}.')
    times = df['time_total_s']
    print(f'Total time elasped: {times.sum():.2f}s (ignoring concurrency).')
    print(f'time per trial: {times.mean():.2f}s.')
    print(f'Date of latest trial: {sorted_df["date"].tolist()[len(df) - 1]}')

    # import pdb
    # pdb.set_trace()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=('Get best run from Ray Tune.'))
    # parser.add_argument('--dataset', type=str, help='dataset')
    # parser.add_argument('--transforms', type=str, help='transforms')
    # parser.add_argument('--exp', type=int, help='exp')
    # parser.add_argument('depth', type=int, help='depth')
    # args = parser.parse_args()
    main()
