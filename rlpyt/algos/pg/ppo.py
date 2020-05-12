
import torch
import higher
from collections import namedtuple

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.algos.utils import (discount_return, generalized_advantage_estimation,
    valid_from_done)
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])
IrOptInfo = namedtuple("IrOptInfo", ["loss", "gradNorm", "ireward", "entropy", "perplexity", "r_loss", "r_gradNorm", "rv_gradNorm"])

class PPO(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """

    def __init__(
            self,
            discount=0.99,
            learning_rate=3e-4,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.0,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=0.5,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
            init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity


class PPO_IM(PPO):

    def __init__(
            self,
            discount=0.9,
            ir_discount=0.99,
            learning_rate=3e-4,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            ):
        """Saves input settings."""
        self.intrinsic_ratio = 1.  # 0.5
        self.intrinsic_discount = ir_discount
        self.intrinsic_reward=True
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.opt_info_fields = tuple(f for f in IrOptInfo._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Extends base ``initialize()`` to initialize optimizer for reward
        """
        super().initialize(agent, n_itr, batch_spec, mid_batch_reset=mid_batch_reset,
            examples=examples, world_size=world_size, rank=rank)
        self.r_optimizer = self.OptimCls(agent.r_model.parameters(),
                lr=self.learning_rate/3., **self.optim_kwargs)
        self.rv_optimizer = self.OptimCls(agent.rv_model.parameters(),
                lr=self.learning_rate/3., **self.optim_kwargs)
        params = list(self.agent.model.parameters())
        self.log_std_params = params[0]
        self.mu_params = params[1:7]
        self.gm = []
        self.gv = []
        # for p in params:
        #    self.gm.append(torch.zeros_like(p, device=self.agent.device))
        #    self.gv.append(torch.zeros_like(p, requires_grad=False, device=self.agent.device))

    def process_returns(self, samples):
        ereturn_, e_advantage, valid = super().process_returns(samples, intrinsic=self.intrinsic_ratio)
        obs, act = buffer_to((samples.env.observation, samples.agent.action), device=self.agent.device)
        ireward = self.agent.r(obs,act).cpu()
        reward, done, value, bv = (samples.env.reward, samples.env.done,
            samples.agent.agent_info.r_value, samples.agent.bootstrap_r_value,)
        done = done.type(reward.dtype)

        with torch.no_grad():
            if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
                return_ = discount_return(reward, done, bv, self.intrinsic_discount)
                advantage = return_ - value
            else:
                advantage, return_ = generalized_advantage_estimation(
                    reward, value, done, bv, self.intrinsic_discount, self.gae_lambda)

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return ereturn_, e_advantage, return_, advantage, valid, ireward

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, i_return, i_advantage, valid, ireward = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        r_loss_inputs = LossInputs(
                agent_inputs=agent_inputs,
                action=samples.agent.action,
                return_=i_return,
                advantage=i_advantage,
                valid=valid,
                old_dist_info=samples.agent.agent_info.dist_info,
                )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = IrOptInfo(*([] for _ in range(len(IrOptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                # self.optimizer.zero_grad()
                # self.rv_optimizer.zero_grad()
                # self.r_optimizer.zero_grad()
                # self.agent.temp_model.zero_grad()
                self.agent.zero_grad()
                # with higher.innerloop_ctx(self.agent.model, self.optimizer) as (fmodel, diffopt):
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)  # , model=fmodel)
                loss.backward(retain_graph=True)
                # diffopt.step(loss)
                if len(self.gm) == 0:
                    print("Create zeros")
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            self.gm.append(torch.zeros_like(p.grad, device=self.agent.device))
                            self.gv.append(torch.zeros_like(p.grad, requires_grad=False, device=self.agent.device))
                # for m in self.gm:
                #    print(m.size())


                # r_loss, rv_loss = self.r_loss(*r_loss_inputs[T_idxs, B_idxs], rnn_state)  # , model=fmodel)
                # rv_loss.backward()
                # r_loss.backward()

                #############################################
                beta1, beta2 = self.optimizer.defaults['betas']
                p_groups = self.optimizer.param_groups
                # print(p_groups)
                temp_params = self.agent.temp_model.parameters()
                variables_to_del = []
                for p, np, m, v in zip(p_groups[0]["params"], temp_params, self.gm, self.gv):
                    # for p in group["params"]:
                    grad = p.grad
                    # state = self.optimizer.state[p]
                    # if len(state) == 0:
                    #     m = buffer_to(torch.zeros_like(grad), device=self.agent.device)
                    #     v = buffer_to(torch.zeros_like(grad), device=self.agent.device)
                    # else:
                    #     m = buffer_to(state["exp_avg"], device=self.agent.device)
                    #     v = buffer_to(state["exp_avg_sq"], device=self.agent.device)
                    # print(m.size(), grad.size())
                    m = m + (grad - m) * (1 - beta1)
                    # with torch.no_grad():
                    v = (v + (grad**2 - v) * (1 - beta2)).detach()
                    # variables_to_del.extend([m, v, grad])
                    np = p.clone().detach() - m * self.learning_rate / (torch.sqrt(v) + 1e-5)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                r_loss, rv_loss = self.r_loss(
                        *r_loss_inputs[T_idxs, B_idxs], rnn_state)
                rv_loss.backward()
                r_loss.backward()  # retain_graph=True)
                # lvd = len(variables_to_del)
                # for i in reversed(range(lvd)):
                #     del variables_to_del[i]
                # print(variables_to_del)
                # exit()
                ############################################
                for p in self.agent.r_model.parameters():
                    p.grad /= (len(idxs)**2)
                r_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.agent.r_model.parameters(), self.clip_grad_norm)
                rv_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.agent.rv_model.parameters(), self.clip_grad_norm)
                self.r_optimizer.step()
                self.rv_optimizer.step()
                # for m in self.gm:
                #     m.grad.zero_()

                opt_info.r_loss.append(r_loss.item())
                opt_info.r_gradNorm.append(r_grad_norm)
                opt_info.rv_gradNorm.append(rv_grad_norm)
                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                opt_info.ireward.append(torch.mean(ireward).item())
                self.update_counter += 1


        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr
        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
            init_rnn_state=None):  # , model=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        # if model is None:
        #     model = self.agent.model
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            mu, log_std, value, _rnn_state = buffer_to(self.agent.model(*agent_inputs, init_rnn_state), device="cpu")
            # dist_info, value, _rnn_state = agent(*agent_inputs, init_rnn_state)
        else:
            mu, log_std, value = buffer_to(self.agent.model(*agent_inputs), device="cpu")
            # dist_info, value = agent(*agent_inputs)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss
        # if torch.isnan(loss):
        #     import pdb
        #     pdb.set_trace()

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity

    def r_loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
            init_rnn_state=None):  # , model=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        # if model is None:
        #     model = self.agent.model
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            mu, log_std, _, _rnn_state = buffer_to(self.agent.temp_model(*agent_inputs, init_rnn_state), device="cpu")
            # dist_info, _, _rnn_state = agent(*agent_inputs, init_rnn_state)
        else:
            mu, log_std, _ = buffer_to(self.agent.temp_model(*agent_inputs), device="cpu")
            # dist_info, _ = agent(*agent_inputs)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        value = self.agent.r_val(agent_inputs.observation, buffer_to(action, device=self.agent.device)).cpu()
        dist = self.agent.temp_distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        # loss = pi_loss + value_loss

        return pi_loss, value_loss

