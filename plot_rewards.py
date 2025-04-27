import numpy as np
import matplotlib.pyplot as plt
import os
import glob


episode_rewards = np.load('rewards_final_ac/episode_rewards, gamma_0.95_lr_5e-05.npy')
window_size = 500
smoothed_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')

plt.figure()
plt.plot(smoothed_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards over Time')
plt.tight_layout()
plt.savefig('episode_rewards_plot_ac.png')
plt.show()


