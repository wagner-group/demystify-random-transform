from adv.models.edsr import EDSR, get_weights
from kornia.augmentation import (RandomEqualize, RandomErasing,
                                 RandomGrayscale, RandomHorizontalFlip,
                                 RandomMotionBlur, RandomSolarize,
                                 RandomVerticalFlip)
from kornia.filters import (BoxBlur, GaussianBlur2d, Laplacian, MedianBlur,
                            Sobel)
from torchvision.transforms import RandomResizedCrop

from .jpeg import DiffJPEG
from .utils import EPS


def init_horizontal_flip(hparams, same_on_batch=False):
    return RandomHorizontalFlip(p=1., same_on_batch=same_on_batch)


def init_vertical_flip(hparams, same_on_batch=False):
    return RandomVerticalFlip(p=1., same_on_batch=same_on_batch)


def init_filter_box(hparams, same_on_batch=False):
    kernel_size = hparams.get('kernel_size', 3)
    return BoxBlur((kernel_size, kernel_size))


def init_filter_gauss(hparams, same_on_batch=False):
    kernel_size = hparams.get('kernel_size', 3)
    sigma = hparams.get('sigma', 2.5)
    return GaussianBlur2d((kernel_size, kernel_size), (sigma, sigma))


def init_filter_median(hparams, same_on_batch=False):
    return MedianBlur((hparams['kernel_size'], hparams['kernel_size']))


def init_filter_motion(hparams, same_on_batch=False):
    return RandomMotionBlur(hparams['kernel_size'], hparams['angle'],
                            hparams['direction'], p=1., resample='bilinear',
                            same_on_batch=same_on_batch)


def init_filter_laplacian(hparams, same_on_batch=False):
    return Laplacian(hparams['kernel_size'])


def init_filter_sobel(hparams, same_on_batch=False):
    return Sobel()


def init_crop(hparams, same_on_batch=False):
    input_size = hparams['input_size']
    return RandomResizedCrop((input_size, input_size),
                             scale=(1. - hparams['alpha'] + EPS,
                                    1. + hparams['alpha']),
                             #  same_on_batch=same_on_batch,
                             )


def init_crop_full(hparams, same_on_batch=False):
    input_size = hparams['input_size']
    return RandomResizedCrop((input_size, input_size),
                             scale=(1. - hparams['alpha'] + EPS,
                                    1. + hparams['alpha']),
                             #  same_on_batch=same_on_batch,
                             )


def init_erase(hparams, same_on_batch=False):
    return RandomErasing(p=1., scale=(0.02, max(0.02 + EPS, hparams['alpha'])),
                         same_on_batch=same_on_batch)


def init_edsr(hparams):
    raise NotImplementedError
    weights = get_weights()
    net = EDSR().eval()
    net.load_state_dict(weights)
    return net


def init_equalize(hparams):
    return RandomEqualize(p=1.)


def init_grayscale(hparams):
    return RandomGrayscale(p=1., keepdim=True)


def init_jpeg(hparams, same_on_batch=False):
    input_size = hparams['input_size']
    return DiffJPEG(input_size, input_size, differentiable=True)


def init_solarize(hparams, same_on_batch=False):
    return RandomSolarize(p=1., same_on_batch=same_on_batch)


name_to_init_tf = {
    # Flipping
    'hflip': init_horizontal_flip,
    'vflip': init_vertical_flip,
    # Blur filter
    # 'boxblur_same': init_filter_box,
    # 'gaussblur_same': init_filter_gauss,
    'boxblur': init_filter_box,
    'gaussblur': init_filter_gauss,
    'medblur': init_filter_median,
    'motionblur': init_filter_motion,
    # Edge detection
    'laplacian': init_filter_laplacian,
    'sobel': init_filter_sobel,
    # Etc.
    'crop': init_crop,
    'edsr': init_edsr,
    'equalize': init_equalize,
    'erase': init_erase,
    'grayscale': init_grayscale,
    'jpeg': init_jpeg,
    'jpeg_full': init_jpeg,
    'solarize': init_solarize,
}


def init_transform(tf_name, hparams, same_on_batch=False):
    assert tf_name in name_to_init_tf
    return name_to_init_tf[tf_name](hparams, same_on_batch=same_on_batch)
