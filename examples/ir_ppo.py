
"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm. Can choose between configurations for use of CPU/GPU for sampling
(serial or parallel) and optimization (serial).

Alternating sampler is another option.  For recurrent agents, a different mixin
is required for alternating sampling (see rlpyt.agents.base.py), feedforward agents
remain unaffected.

"""
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.envs.meta import make as meta_make
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.pg.ppo import PPO_IM, PPO
from rlpyt.agents.pg.mujoco import MujocoFfRewAgent, MujocoFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
import torch
import os
import numpy as np


def build_and_train(env_id="Hopper-v3", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2, ir=False):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    elif sample_mode == "gpu":
        Sampler = GpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "alternating":
        Sampler = AlternatingSampler
        affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
        affinity["alternating"] = True  # Sampler will check for this.
        print(f"Using Alternating GPU parallel sampler, {gpu_cpu} for sampling and optimizing.")

    sampler = Sampler(
        EnvCls=gym_make,  # meta_make,
        # TrajInfoCls=AtariTrajInfo,
        env_kwargs=dict(id=env_id),
        batch_T=128,  # 5 time-steps per sampler iteration.
        batch_B=4,  # 16 parallel environments.
        max_decorrelation_steps=400,
        accumulate_reward=40
    )
    algo = PPO_IM(gae_lambda=0.95, clip_grad_norm=0.5, discount=0.9, ir_discount=0.998,
            ratio_clip=0.1, minibatches=32) if ir else PPO(gae_lambda=0.95,
                    clip_grad_norm=0.5, ratio_clip=0.1, minibatches=32)  # Run with defaults.
    agent = MujocoFfRewAgent() if ir else MujocoFfAgent()
    # if ir:
    #     agent.r_model.load_state_dict(torch.load(log_dir + '/reward.pt'))
    #     agent.rv_model.load_state_dict(torch.load(log_dir + '/rvalue.pt'))
    #     # torch.save(agent.r_model.state_dict(), log_dir + '/reward.pt')
    #     # torch.save(agent.rv_model.state_dict(), log_dir + '/rvalue.pt')
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )
    config = dict(env_id=env_id)
    name = "ppo_" + env_id
    if ir:
        log_dir = "/scratch/cluster/ishand/reward/im_inverted_40_discount_09"
    else:
        log_dir = "/scratch/cluster/ishand/reward/inverted_discount_40"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='Meta World Env', default='InvertedPendulum-v2')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
        type=str, default='serial', choices=['serial', 'cpu', 'gpu', 'alternating'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=4)
    parser.add_argument('--ir', help='Use intrinsic reward', action='store_true')
    args = parser.parse_args()
    
    # Set Seed
    seed = args.run_ID * 421
    os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda_idx is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
        ir=args.ir,
    )
