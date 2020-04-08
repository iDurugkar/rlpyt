
import torch

from rlpyt.agents.pg.gaussian import (GaussianPgAgent,
    RecurrentGaussianPgAgent, AlternatingRecurrentGaussianPgAgent)
from rlpyt.agents.pg.base import AgentInfoIr
from rlpyt.agents.base import AgentStep
from rlpyt.models.pg.mujoco_ff_model import MujocoFfModel
from rlpyt.models.pg.mujoco_lstm_model import MujocoLstmModel
from rlpyt.utils.buffer import buffer_to, torchify_buffer
from rlpyt.models.reward.mujoco_im_model import RewardFfModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd


class MujocoMixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    Now supports observation normalization, including multi-GPU.
    """
    _ddp = False  # Sets True if data parallel, for normalized obs

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract observation_shape and action_size."""
        assert len(env_spaces.action.shape) == 1
        return dict(observation_shape=env_spaces.observation.shape,
                    action_size=env_spaces.action.shape[0])

    def update_obs_rms(self, observation):
        observation = buffer_to(observation, device=self.device)
        if self._ddp:
            self.model.module.update_obs_rms(observation)
        else:
            self.model.update_obs_rms(observation)

    def data_parallel(self, *args, **kwargs):
        super().data_parallel(*args, **kwargs)
        self._ddp = True


class MujocoFfAgent(MujocoMixin, GaussianPgAgent):

    def __init__(self, ModelCls=MujocoFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class MujocoFfRewAgent(MujocoFfAgent):

    def __init__(self, ModelCls=MujocoFfModel, RewardCls=RewardFfModel, r_model_kwargs=None, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
        if r_model_kwargs is None:
            r_model_kwargs = dict(hidden_sizes=[64, 64])
        save__init__args(locals())

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        """extends gaussian initialization to build reward function"""
        super().initialize(env_spaces, share_memory,
                global_B=global_B, env_ranks=env_ranks)
        self.r_model = self.RewardCls(**self.env_model_kwargs, **self.r_model_kwargs, rew_nonlinearity=torch.atan)
        self.rv_model = self.RewardCls(**self.env_model_kwargs, **self.r_model_kwargs)
        self.temp_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
        self.temp_distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.r_model.to(self.device)
        self.rv_model.to(self.device)
        self.temp_model.to(self.device)

    def zero_grad(self):
        self.model.zero_grad()
        self.r_model.zero_grad()
        self.rv_model.zero_grad()
        self.temp_model.zero_grad()

    def r(self, observation, action):
        """compute reward for state and taken action"""
        model_inputs = buffer_to(torchify_buffer((observation, action)), device=self.device)
        r = self.r_model(*model_inputs)
        return r

    def update_obs_rms(self, observation):
        super().update_obs_rms(observation)
        observation = buffer_to(observation, device=self.device)
       #  self.r_model.update_obs_rms(observation)

    def r_val(self, observation, action):
        """compute value for external reward function"""
        model_inputs = buffer_to(torchify_buffer((observation, action)), device=self.device)
        v = self.rv_model(*model_inputs)
        return v
    
    def call_temp_model(self, observation, prev_action, prev_reward):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, value = self.temp_model(*model_inputs)
        return buffer_to((DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """call super and then add reward information"""
        agent_step = super().step(observation, prev_action, prev_reward)
        reward = self.r(observation, agent_step.action).cpu()
        r_value = self.r_val(observation, agent_step.action).cpu()
        agent_info = AgentInfoIr(*agent_step.agent_info, reward=reward, r_value=r_value)
        return AgentStep(action=agent_step.action, agent_info=agent_info)


class MujocoLstmAgent(MujocoMixin, RecurrentGaussianPgAgent):

    def __init__(self, ModelCls=MujocoLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class AlternatingMujocoLstmAgent(MujocoMixin,
        AlternatingRecurrentGaussianPgAgent):

    def __init__(self, ModelCls=MujocoLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
