
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel

class RMlpModel(torch.nn.Module):
    """
    Model that takes as input a state and generates reward
    """
    def __init__(
            self,
            observation_shape,
            hidden_sizes,
            action_size,    # =None,  # Unused but accept kwarg.
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.mlp = MlpModel(
            input_size=int(np.prod(observation_shape)) + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )

    def forward(self, observation, action):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        r_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        r = self.mlp(r_input).squeeze(-1)
        r = restore_leading_dims(r, lead_dim, T, B)
        return r 

    def load(self, path):
        self.load_state_dict(torch.load(path))
