import torch
from torch import nn
import numpy as np
from copy import deepcopy
import subprocess
import os, shutil
import time

N = 40
T = 5
G = 20

base_params = torch.load('/scratch/cluster/ishand/reward/meta_params.pt')
base_dir = '/scratch/cluster/ishand/results/reward_meta'
gen_name = os.path.join(base_dir, 'gen0')
if os.path.isdir(gen_name):
    print('files exist. Scorching earth!')
    shutil.rmtree(base_dir)

agent_params = [deepcopy(base_params) for _ in range(N)]
for i, ap in enumerate(agent_params):
    for tensor in ap:
        k = ap[tensor].size()[-1]
        nn.init.uniform_(ap[tensor], a=-np.sqrt(k), b=np.sqrt(k))
    dname = os.path.join(gen_name, 'run_%d' % i)
    os.makedirs(dname)
    torch.save(ap, os.path.join(dname, 'params.pt'))

print('Saved params')

for g in range(G):
    gen_name = os.path.join(base_dir, 'gen{}'.format(g))
    results = []
    subprocess.run(['condorify_gpu', 'run_train.sh', gen_name, str(N), 'gen_{}'.format(g)],
                  stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
    print('started gen {}'.format(g))
    finished = False
    checked = 0
    while not finished:
        time.sleep(60)
        out = subprocess.run(['condor_q'], capture_output=True, text=True)
        arr = out.stdout.split('\n')
        jobs_left = int(arr[-2].split()[0])
        print('Checking... {} jobs still running'.format(jobs_left))
        if jobs_left == 0:
            finished = True
        checked += 1
        if checked > 240:
            print('Jobs running more than 4 hours. Killing')
            subprocess.run(['condor_rm', 'ishand'], capture_output=True, text=True)
    for anum in range(N):
        # subprocess.run(['python', 'train.py', '--agent_id', '{}'.format(anum), '--log_dir', gen_name],
        #               stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        with open(os.path.join(gen_name, 'run_{}'.format(anum), 'debug.log'), 'r') as fr:
            try:
                avg_ret = float(fr.readlines()[-1].split()[-1])
                print(avg_ret)
                results.append(avg_ret)
            except Exception:
                print("couldn't read {}".format(anum))
    print('End of Generation')
    t_best = np.argsort(results)[-T:]
    print('Best in gen:')
    print(t_best)
    print(np.asarray(results)[t_best])
    print('********************************')
    for n in range(N-3):
        pidx = np.random.choice(t_best)
        agent_params = torch.load(os.path.join(gen_name, 'run_{}'.format(pidx), 'params.pt'))
        for tensor in agent_params:
            agent_params[tensor] += torch.randn_like(agent_params[tensor]) * 0.01
            # torch.Tensor(np.random.normal(size=agent_params[tensor].size()) * 0.01, dtype=agent_params[tensor].type())
        dname = os.path.join(base_dir, 'gen{}'.format(g+1), 'run_%d' % n)
        os.makedirs(dname)
        torch.save(agent_params, os.path.join(dname, 'params.pt'))
    print('Created new generation')
    for i in range(3):
        dname = os.path.join(base_dir, 'gen{}'.format(g+1), 'run_%d' % (N - 1 - i))
        os.makedirs(dname)
        agent_params = torch.load(os.path.join(gen_name, 'run_{}'.format(t_best[-1-i]), 'params.pt'))
        torch.save(agent_params, os.path.join(dname, 'params.pt'))
print("End of Code")

