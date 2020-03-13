
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

print('import')
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.meta import make as meta_make
print('done import')
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
import torch


def build_and_train(env_id="sweeper", run_ID=0, cuda_idx=0):
    print('start')
    sampler = SerialSampler(
        EnvCls=meta_make,
        env_kwargs=dict(env_id=env_id),
        eval_env_kwargs=dict(env_id=env_id),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )
    print('sampler ready')
    algo = SAC()  # Run with defaults.
    print('algo created')
    agent = SacAgent()
    print('agent_created')
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )
    print('runner created')
    config = dict(env_id=env_id)
    name = "sac_" + env_id
    log_dir = "/scratch/cluster/ishand/reward/meta_test_1e6"
    print('starting train')
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    print('gonna parse args')
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='other')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=6)
    args = parser.parse_args()
    print('setting seed')
    torch.manual_seed(args.run_ID * 42)
    print('call train?')
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
    )
