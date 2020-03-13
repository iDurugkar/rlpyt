
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.envs.meta import make as meta_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from os.path import join


def build_and_train(env_id="sweeper", run_id=0, log_dir='/scratch/cluster/ishand/reward/meta_test', cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=meta_make,  # gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )
    algo = SAC()  # Run with defaults.
    agent = SacAgent(r_path=join(log_dir, 'run_%d' % run_id, 'params.pt'))
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e5,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    name = "sac_" + env_id
    with logger_context(log_dir, run_id, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='sweeper')
    parser.add_argument('--agent_id', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--log_dir', help='directory to store results', type=str, default='/scratch/cluster/ishand/reward/meta_test')
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_id=args.agent_id,
        log_dir=args.log_dir,
        cuda_idx=args.cuda_idx,
    )
