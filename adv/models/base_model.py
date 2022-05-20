import torch
from torch.nn import Module


class BaseModel(Module):

    def load_weights(self, save_path):
        self.load_state_dict(torch.load(save_path))

    def save_weights(self, save_path):
        torch.save(self.state_dict(), save_path)
