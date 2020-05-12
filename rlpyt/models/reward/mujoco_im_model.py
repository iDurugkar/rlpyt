
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.running_mean_std import RunningMeanStdModel



class IPRewardModel(torch.nn.Module):
    """
    Model used in Mujoco Inverted Pendulum: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLP for state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            rew_nonlinearity=torch.nn.Tanh,  # Module form.
            init_log_std=0.,
            normalize_observation=False,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
            ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        sequence = list()
        sequence.extend([torch.nn.Linear(input_size, input_size), rew_nonlinearity()])
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, observation, action):
        """
        Compute reward and externalvalue estimate from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)

        obs_flat = observation.view(T * B, -1)
        r_input = obs_flat  # torch.cat([obs_flat, a_flat], dim=1)
        r = self.model(r_input)
        r = ((1. - torch.abs(obs_flat)) * r).sum(-1)
        r = restore_leading_dims(r, lead_dim, T, B)
        return r

    def update_obs_rms(self, observation):
        return


class RewardFfModel(torch.nn.Module):
    """
    Model commonly used in Mujoco locomotion agents: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLP for state-value estimate.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            rew_nonlinearity=None,  # Module form.
            init_log_std=0.,
            normalize_observation=False,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
            ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape)) + action_size
        hidden_sizes = hidden_sizes or [64, 64]
        mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            nonlinearity=hidden_nonlinearity,
        )
        self.rew_nonlinearity = rew_nonlinearity
        # if rew_nonlinearity is not None:
        #     print("Adding nonlinearity to output")
        #     self.mlp = torch.nn.Sequential(mlp, rew_nonlinearity())
        # else:
        self.mlp = mlp
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation

    def forward(self, observation, action):
        """
        Compute reward and externalvalue estimate from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)

        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        obs_flat = observation.view(T * B, -1)
        a_flat = action.view(T*B, -1)
        r_input = torch.cat([obs_flat, a_flat], dim=1)
        r = self.mlp(r_input).squeeze(-1)
        if self.rew_nonlinearity is not None:
            r = self.rew_nonlinearity(r)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        r = restore_leading_dims(r, lead_dim, T, B)
        return r

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)
