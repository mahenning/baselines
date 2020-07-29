import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle


matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 18})
os_path = '/mnt/SSD/Marko/Dokumente/Uni/SoSe20/Minecraft/baselines/results/big_epochs/'
exp_name = 'LunarLander-v2'
env_path = exp_name+'-1_512_10000_512.pickle'

with (open(os_path+env_path, 'rb')) as file:
    data = pickle.load(file)
    file.close()

data_mean = [np.mean(val) for val in data]
data_by_bin = list(map(list, zip(*data)))
data_by_bin_mean = [np.mean(val) for val in data_by_bin]
flattened = [val for sublist in data for val in sublist]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 24))
ax1.plot(range(len(flattened)), flattened)
# ax1.title.set_text('Number of clamping over the whole training', size=20)
ax1.set(title='Number of clamping over the whole training',
        xlabel='Number of training steps*epoch*batch', ylabel='Number of clampings')
ax2.plot(range(0, len(data_mean)), data_mean)
ax2.set(title='Number of clamping, mean values of epoch+batch',
        xlabel='Number of training steps', ylabel='Number of clampings')
ax3.plot(range(len(data_by_bin_mean)), data_by_bin_mean)
ax3.set(title='Number of clamping, mean over all episode steps',
        xlabel='Position in epoch+batch', ylabel='Number of clampings')
ax3.xaxis.set_major_locator(MaxNLocator(16, integer=True))
# plt.show()
fig.suptitle(exp_name)
fig.savefig(os_path+env_path[:-7])
