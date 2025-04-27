import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import ClipAction

from model import VPG

obs_mean = None
obs_var = None
obs_count = None

def normalize(x):
    global obs_mean, obs_var, obs_count
    obs_count += 1
    delta = x - obs_mean
    obs_mean += delta / obs_count
    obs_var += delta * (x - obs_mean)
    return (x - obs_mean) / (np.sqrt(obs_var / obs_count) + 1e-8)

def evaluate(model_path, norm_path, env_id='Pusher-v5', episodes=5, max_steps=200):
    global obs_mean, obs_var, obs_count
    # Load normalization arrays
    data = np.load(norm_path)
    obs_mean = data['mean']
    obs_var = data['var']
    obs_count = data['count']

    # Prepare environment
    base_env = gym.make(env_id, render_mode='human')
    env = ClipAction(base_env)

    # Build policy network and load weights
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = VPG(obs_dim=obs_dim, act_dim=act_dim)
    state_dict = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    policy.eval()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            obs_norm = normalize(obs)
            with torch.no_grad():
                mean, _ = policy(torch.tensor(obs_norm, dtype=torch.float32)).chunk(2, dim=-1)
            action = mean.numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, truncated, info = env.step(action)
            env.render()
            total_reward += reward

            if done or truncated:
                break

        print(f"Episode {ep} – Reward: {total_reward:.2f}")

    env.close()


if __name__ == '__main__':

    model_path = "models_final_ac/ac_pusher_50000.pt"
    norm_path = "models_final_ac/obs_rms.npz"
    episodes = 100

    evaluate(model_path, norm_path, episodes=episodes)
