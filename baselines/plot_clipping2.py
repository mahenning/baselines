import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import tensorflow as tf


matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 18})
os_path = '/mnt/SSD/Marko/Dokumente/Uni/SoSe20/Minecraft/baselines/results/big_epochs/'
exp_name = 'LunarLander-v2'
env_path = exp_name+'-1_512_10000_512.pickle'

with (open(os_path+env_path, 'rb')) as file:
    data = pickle.load(file)
    file.close()

data = data[0]
data = [tf.cast(x, tf.int32).numpy() for x in data]
data2 = np.sum(data, 0)
data3 = np.sum(data, 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 24))
ax1.plot(range(len(data2)), data2)
ax2.plot(range(len(data3)), data3)
plt.show()
