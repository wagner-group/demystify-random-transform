import torch
from torch.nn import Module


class Wrapper(Module):

    def __init__(self):
        super().__init__()

    def load_weights(self, save_path):
        # self.base_net.load_weights(save_path)
        try:
            self.load_state_dict(torch.load(save_path))
        except RuntimeError as e:
            # print(e)
            print('Loading weights to base model only...')
            self.base_net.load_weights(save_path)

    def save_weights(self, save_path):
        # self.base_net.save_weights(save_path)
        torch.save(self.state_dict(), save_path)

    def get_orig_base_net(self):
        if isinstance(self.base_net, Wrapper):
            return self.base_net.get_orig_base_net()
        else:
            return self.base_net

    def check_network_type(self, types):
        """Check type of `net`.

        Args:
            types (tuple or object): Types of object to check against

        Returns:
            bool: Whether `self.base_net` has the same type as `types`.
        """
        return isinstance(self, types)
