import numpy as np
import matplotlib.pyplot as plt

rews = np.load("rews.npy")
plt.plot(rews, label='Cumulative Reward')
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards over Episodes (Actor Critic, GAE)")
plt.grid()

# Calculate the moving average
window_size = 500
moving_avg = np.convolve(rews, np.ones(window_size)/window_size, mode='valid')
plt.plot(np.arange(window_size-1, len(rews)), moving_avg, color='red', label='Moving Average')
plt.legend()

plt.savefig("cumulative_reward.png")
plt.show()

