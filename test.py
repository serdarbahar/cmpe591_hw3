# test_vpg_pusher.py

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import ClipAction

from model import VPG

# ─── load the same obs‐rms you saved ─────────────────────────────
rms = np.load('models_final_v1/obs_rms.npz')
obs_mean, obs_var, obs_count = rms['mean'], rms['var'], rms['count']

def normalize(x):
    # this must match exactly your training normalize()
    global obs_mean, obs_var, obs_count
    obs_count += 1
    delta = x - obs_mean
    obs_mean += delta / obs_count
    obs_var += delta * (x - obs_mean)
    return (x - obs_mean) / (np.sqrt(obs_var / obs_count) + 1e-8)


def evaluate(model_path, env_id='Pusher-v5', episodes=5, max_steps=200):
    # 1. Make the env with human rendering, then wrap for action clipping
    DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    #"distance": 4.0,
    "distance": 3.0,
    "azimuth": 135.0,
    "elevation": -22.5,
}
    base_env = gym.make(env_id,default_camera_config=DEFAULT_CAMERA_CONFIG, render_mode='human')
    env = ClipAction(base_env)

    # 2. Reconstruct your policy network and load weights
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = VPG(obs_dim=obs_dim, act_dim=act_dim)
    state_dict = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    policy.eval()

    for ep in range(1, episodes+1):
        obs, _ = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            # forward pass to get mean action
            obs_norm = normalize(obs)
            with torch.no_grad():
                mean, _ = policy(torch.tensor(obs_norm, dtype=torch.float32)).chunk(2, dim=-1)
            action = mean.numpy()

            # just in case: clip into valid bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # step + render
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            total_reward += reward

            if done or truncated:
                break

        print(f"Episode {ep} – Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # Path to the model you want to test
    model_path = "models_final_v1/vpg_pusher_984000.pt"
    episodes = 100
    evaluate(model_path, episodes=episodes)
  
#980000
#988000