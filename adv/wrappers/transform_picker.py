import numpy as np

from ..utils import set_temp_seed


class TransformPicker(object):
    def __init__(self, size, n_transforms, n_subset, tf_order='random',
                 n_samples=4, group_size=10, seed=None):
        """TransformPicker is an iterator that tells RandWrapper which 
        transformation to apply and to which samples in the batch. Most of the
        transforms exploit parallelization on GPUs, but each sample should see
        different permutations of the transforms. When tf_order is 'random',
        TransformPicker groups samples that should see the same type of 
        transform to exploit at least partial parallelization.

        Args:
            size (int): Batch size
            n_transforms (int): Number of the total transformation types (K)
            n_subset (int): Number of the transforms to apply per samples (S)
            tf_order (str, optional): Permutation method for the transforms 
                (options: 'random', 'fixed', 'group') Defaults to 'random'.
            group_size (int, optional): Used with 'group' tf_order. Specify the
                number of samples in a batch to be grouped together and see the
                same permutation of the transformations. Defaults to 10.
            seed (int, optional): Random seed. Defaults to None.
        """
        assert tf_order in ('random', 'fixed', 'group', 'ens')
        self.tf_order = tf_order
        self.transform_idxs = {}
        self.counter = 0
        self.size = size
        self.group_size = group_size
        self.n_samples = n_samples
        self.n_subset = min(n_subset, n_transforms)
        self.n_transforms = n_transforms
        if seed is not None:
            with set_temp_seed(seed):
                self._init_order()
        else:
            self._init_order()

    def _init_order(self):
        if self.tf_order == 'random':
            # Get transform order for each sample in the batch
            for i in range(self.size):
                im_transform_idxs = list(np.random.choice(
                    range(self.n_transforms), self.n_subset, replace=False))
                self.transform_idxs[i] = im_transform_idxs
        elif self.tf_order in ('group', 'ens'):
            # An ordered subset of size `group_size` of an expanded batch gets
            # the same order of transforms. `group_size` should be set to the
            # original batch size.
            if self.tf_order == 'group':
                self.num_groups = int(np.ceil(self.size / self.group_size))
            else:
                self.num_groups = self.n_samples
                self.group_size = int(self.size / self.n_samples)   # should be orig batch size
                assert self.group_size * self.n_samples == self.size
            self.group_counter = 0
            self.transform_order = []
        else:
            # Initialize transform_order once in the begining for 'fixed'
            self.transform_order = list(np.random.choice(
                range(self.n_transforms), self.n_subset, replace=False))
        # TODO: Implement random group order

    def __iter__(self):
        return self

    def __next__(self):
        if self.tf_order == 'random':
            if len(self.transform_idxs) == 0:
                raise StopIteration
            counts = {}
            # Get counts of transforms to apply for each image
            for im_idx in self.transform_idxs:
                idx = self.transform_idxs[im_idx][0]
                if idx not in counts:
                    counts[idx] = []
                counts[idx].append(im_idx)

            # Get the most common transform and corresponding image indices
            idx, im_idxs = max(counts.items(), key=lambda x: len(x[1]))

            # Pop the transform to be applied
            for im_idx in im_idxs:
                self.transform_idxs[im_idx].pop(0)
                if len(self.transform_idxs[im_idx]) == 0:
                    del self.transform_idxs[im_idx]
        elif self.tf_order in ('group', 'ens'):
            # Each group uses n_subset transforms
            if self.counter >= self.n_subset * self.num_groups:
                raise StopIteration
            if self.counter % self.n_subset == 0:
                self.group_counter += 1
                self.transform_order = list(np.random.choice(
                    range(self.n_transforms), self.n_subset, replace=False))
                self.im_idxs = np.arange((self.group_counter - 1) * self.group_size,
                                         self.group_counter * self.group_size)
            idx = self.transform_order[self.counter % self.n_subset]
            im_idxs = self.im_idxs
            self.counter += 1
        else:
            if self.counter >= self.n_subset:
                raise StopIteration
            idx = self.transform_order[self.counter]
            im_idxs = np.arange(self.size)
            self.counter += 1

        return idx, im_idxs
