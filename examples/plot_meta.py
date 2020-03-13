import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 14})

results = []
base_dir = '/scratch/cluster/ishand/results/reward_27'

plt.figure(1)
for g in range(20):
    pop = []
    print('gen %d' % g)
    for r in range(40):
        with open(os.path.join(base_dir, 'gen{}'.format(g), 'run_{}'.format(r), 'debug.log'), 'r') as fr:
            try:
                avg_ret = float(fr.readlines()[-1].split()[-1])
                pop.append(avg_ret)
                # plt.plot(g, avg_ret, 'o')
            except Exception:
                print('Exception. last lines are:')
                print(fr.readlines()[-5:])
    results.append(pop)
plt.boxplot(results)
plt.title('InvertedPendulum')
plt.xlabel('Generation')
plt.ylabel('Avg Return')
plt.tight_layout()
plt.savefig('/scratch/cluster/ishand/results/reward_27/results.png')
plt.close()

