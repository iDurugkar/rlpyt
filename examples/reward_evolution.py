import torch
from torch import nn
import numpy as np
from copy import deepcopy
import subprocess
import os

N = 30
T = 5
G = 20

base_params = torch.load('/scratch/cluster/ishand/reward/test.pt')
base_dir = '/scratch/cluster/ishand/reward'
gen_name = os.path.join(base_dir, 'gen0')

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
    for anum in range(N):
        subprocess.run(['python', 'train.py', '--agent_id', '{}'.format(anum), '--log_dir', gen_name],
                      stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
        with open(os.path.join(gen_name, 'run_{}'.format(anum), 'debug.log'), 'r') as fr:
            avg_ret = float(fr.readlines()[-1].split()[-1])
            print(avg_ret)
            results.append(avg_ret)
    print('End of Generation')
    t_best = np.argsort(results)[-T:]
    for n in range(N-1):
        pidx = np.random.choice(t_best)
        agent_params = torch.load(os.path.join(gen_name, 'run_{}'.format(pidx), 'params.pt'))
        for tensor in agent_params:
            agent_params[tensor] += torch.Tensor(np.random.normal(size=agent_params[tensor].size()) * 0.01, dtype=agent_params[tensor].type())
        dname = os.path.join(base_dir, 'gen{}'.format(g+1), 'run_%d' % n)
        os.makedirs(dname)
        torch.save(agent_params, os.path.join(dname, 'params.pt'))
    dname = os.path.join(base_dir, 'gen{}'.format(g+1), 'run_%d' % (N-1))
    os.makedirs(dname)
    agent_params = torch.load(os.path.join(gen_name, 'run_{}'.format(t_best[-1]), 'params.pt'))
    torch.save(agent_params, os.path.join(dname, 'params.pt'))
print("End of Code")

