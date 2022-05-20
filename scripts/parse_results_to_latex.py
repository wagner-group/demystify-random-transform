import pickle
from os import listdir
from os.path import isfile, join

# path = '/home/chawin/rand-smooth/save/imagenette/resnet34/rand/affine-colorjitter-erase-fft-gamma-gaussblur-hflip-jpeg-laplacian-medblur-motionblur-poisson-precision-salt-sharp-sobel-solarize-vflip/1/'
path = '/home/chawin/rand-smooth/save/cifar10/resnet/rand/affine-colorjitter-crop-erase-gamma-gray-gray1-graymix-hflip-hsv-jpeg-normal-precision-salt-sharp-sobel-solarize-speckle-uniform-vflip/1/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
for f in onlyfiles:
    if '-nd.pkl' in f or '-ns.pkl' in f:
        out = pickle.load(open(join(path, f), 'rb'))
        num_exp = 3
    elif '.pkl' in f:
        out = pickle.load(open(join(path, f), 'rb'))
        num_exp = 6
    else:
        continue

    if len(out) == 0:
        print(f)
        continue

    acc = []
    conf = []
    # print(f, out)
    try:
        for i in range(num_exp):
            if 'adv_0' in out:
                acc.append(out['adv_0'][f'out_{i}']['acc_mean'])
                conf.append(out['adv_0'][f'out_{i}']['conf_int'][1])
            elif 'out_0' in out:
                acc.append(out[f'out_{i}']['acc_mean'])
                conf.append(out[f'out_{i}']['conf_int'][1])
            elif 'adv' in out:
                acc.append(out['adv'][f'out_{i}']['acc_mean'])
                conf.append(out['adv'][f'out_{i}']['conf_int'][1])
    except:
        print(f)
        print(out)
        continue

    # import pdb
    # pdb.set_trace()
    message = f'{f}:'
    for i in [3, 4, 5, 0, 1, 2]:
        message += f' ${acc[i]:.2f} \pm {conf[i]:.2f}$ &'
    print(message)
