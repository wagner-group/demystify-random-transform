import numpy as np

transforms = ['affine', 'boxblur', 'colorjitter', 'crop', 'drop_pixel',
              'erase', 'fft', 'gamma', 'gaussblur', 'grayscale', 'hflip',
              'laplacian', 'jpeg', 'medblur', 'motionblur', 'normal',
              'poisson', 'posterize', 'salt', 'sharp', 'sobel', 'solarize',
              'speckle', 'swirl', 'uniform', 'vflip']
np.random.shuffle(transforms)
print(sorted(transforms[:16]))
