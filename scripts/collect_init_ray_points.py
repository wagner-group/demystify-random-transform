import pdb
import pickle

from ray.tune import Analysis
from torchvision import transforms


def main(dataset, tf_all, tf_dict):

    tf_all_split = tf_all.split('-')

    init_points = []
    for transforms in tf_dict:
        for exp in tf_dict[transforms]:
            if dataset == 'cifar':
                analysis = Analysis(
                    f'./save/cifar10/resnet/rand/{transforms}/{exp}/tune/')
            else:
                analysis = Analysis(
                    f'./save/imagenette/resnet34/rand/{transforms}/{exp}/tune/')
            df = analysis.dataframe()

            for trial in range(len(df)):
                config = {}
                for tf in tf_all_split:
                    if tf in transforms:
                        config[tf] = df[f'config/{tf}'][trial]
                    else:
                        config[tf] = 0
                metric = df['weight_acc'][trial]
                assert isinstance(metric, float)
                init_points.append({'config': config,
                                    'metric': metric})
    pickle.dump(init_points, open('save/cifar_rand10_ray_init.pkl', 'wb'))
    print(len(init_points))


if __name__ == '__main__':

    # dataset = 'imagnette'
    dataset = 'cifar'

    tf_all = 'affine-boxblur-colorjitter-crop-erase-fft-gamma-gaussblur-gray-gray1-gray2-graymix-hflip-hsv-jpeg-lab-laplacian-medblur-motionblur-normal-pepper-poisson-precision-salt-sharp-sobel-solarize-speckle-swirl-uniform-vflip-xyz-yuv'

    transforms = [
        # All
        'affine-boxblur-colorjitter-crop-erase-fft-gamma-gaussblur-gray-gray1-gray2-graymix-hflip-hsv-jpeg-lab-laplacian-medblur-motionblur-normal-pepper-poisson-precision-salt-sharp-sobel-solarize-speckle-swirl-uniform-vflip-xyz-yuv',
        # No noise
        'affine-boxblur-colorjitter-crop-fft-gamma-gaussblur-gray-gray1-gray2-graymix-hflip-hsv-jpeg-lab-laplacian-medblur-motionblur-precision-sharp-sobel-solarize-swirl-vflip-xyz-yuv',
        # No blur
        'affine-colorjitter-crop-erase-fft-gamma-gray-gray1-gray2-graymix-hflip-hsv-jpeg-lab-laplacian-normal-pepper-poisson-precision-salt-sharp-sobel-solarize-speckle-swirl-uniform-vflip-xyz-yuv',
        # No color space
        'affine-boxblur-colorjitter-crop-erase-fft-gamma-gaussblur-hflip-jpeg-laplacian-medblur-motionblur-normal-pepper-poisson-precision-salt-sharp-sobel-solarize-speckle-swirl-uniform-vflip',
        # No edge detection
        'affine-boxblur-colorjitter-crop-erase-fft-gamma-gaussblur-gray-gray1-gray2-graymix-hflip-hsv-jpeg-lab-medblur-motionblur-normal-pepper-poisson-precision-salt-sharp-solarize-speckle-swirl-uniform-vflip-xyz-yuv',
        # No lossy compression
        'affine-boxblur-colorjitter-crop-erase-gamma-gaussblur-gray-gray1-gray2-graymix-hflip-hsv-lab-laplacian-medblur-motionblur-normal-pepper-poisson-salt-sharp-sobel-solarize-speckle-swirl-uniform-vflip-xyz-yuv',
        # No geometric
        'boxblur-colorjitter-erase-fft-gamma-gaussblur-gray-gray1-gray2-graymix-hsv-jpeg-lab-laplacian-medblur-motionblur-normal-pepper-poisson-precision-salt-sharp-sobel-solarize-speckle-uniform-xyz-yuv',
        # No stylization
        'affine-boxblur-crop-erase-fft-gaussblur-gray-gray1-gray2-graymix-hflip-hsv-jpeg-lab-laplacian-medblur-motionblur-normal-pepper-poisson-precision-salt-sobel-speckle-swirl-uniform-vflip-xyz-yuv',
    ]

    exp = [
        # [26, 27, 28, 29, 30],
        [13],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ]

    tf_dict = {}
    for i, key in enumerate(transforms):
        tf_dict[key] = exp[i]

    main(dataset, tf_all, tf_dict)
