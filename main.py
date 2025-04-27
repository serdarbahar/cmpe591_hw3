import gymnasium as gym
from gymnasium.wrappers import ClipAction
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim
import numpy as np
from tqdm import tqdm
import os

from model import VPG, ValueNet  
from agent import VPG_Agent, AC_Agent  

obs_dim = 23
obs_mean, obs_var, obs_count = np.zeros(obs_dim), np.ones(obs_dim), 1e-4

def normalize(x):
    global obs_mean, obs_var, obs_count
    obs_count += 1
    delta = x - obs_mean
    obs_mean += delta / obs_count
    obs_var += delta * (x - obs_mean)
    return (x - obs_mean) / (np.sqrt(obs_var / obs_count) + 1e-8)
                             
def main(episodes=1000, gamma=0.99, lr=1e-4, variance = 0.5, lam=0.99, value_lr=1e-4):
    env_id = 'Pusher-v5'
    num_episodes = episodes
    max_timesteps = 200
    save_interval = 200  

    base_env = gym.make(env_id)
    env = ClipAction(base_env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    
    ## For policy gradient
    #agent = VPG_Agent(gamma=gamma, LR=lr)
    #agent.model = VPG(obs_dim=obs_dim, act_dim=act_dim)
    #agent.optimizer = optim.Adam(agent.model.parameters(), lr=lr)

    ## For actor-critic
    agent = AC_Agent(gamma=gamma, LR=lr, ValueLR=value_lr, lam=lam, obs_dim=obs_dim, act_dim=act_dim)

    episode_rewards = []

    for ep in tqdm(range(num_episodes), desc=f'Training Episodes, gamma={gamma}, lr={lr}'):
        state, _ = env.reset()
        ep_reward = 0.0
        variance = variance
        lamda = 0.9997
             
        for t in range(max_timesteps):
            state = normalize(state)
            action = agent.decide_action(state, variance)
            next_state, reward, done, truncated, info = env.step(action.numpy())
            #env.render()
            agent.add_reward(reward)
            state = next_state
            ep_reward += reward
            if done or truncated:
                break

        agent.update_model()
        episode_rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            variance = lamda * variance  

        if (ep+1) % 20 == 0:
            tqdm.write(f"Episode {ep + 1}/{num_episodes}, Reward: {ep_reward:.2f}")

        if (ep + 1) % save_interval == 0:
            
            save_folder = "models_final_ac"
            os.makedirs(save_folder, exist_ok=True)
            torch.save(agent.model.state_dict(), os.path.join(save_folder, f"ac_pusher_{ep + 1}.pt"))
           

    env.close()
    episode_rewards = np.array(episode_rewards)
    os.makedirs('rewards_final_ac', exist_ok=True)
    reward_save_path = f"rewards_final_ac/episode_rewards, gamma_{gamma}_lr_{lr}.npy"
    np.save(reward_save_path, episode_rewards)
    print(f"Episode rewards saved to {reward_save_path}")

    observation_save_path = os.path.join(save_folder, f"obs_rms.npz")
    np.savez(
    observation_save_path,
    mean=obs_mean,
    var=obs_var,
    count=obs_count
    )
    print("Saved observation normalization stats to obs_rms.npz")


if __name__ == '__main__':
    gamma = 0.95
    lr = 5e-5
    lam = 0.99
    value_lr = 5e-4
    episodes = 50000
    
    main(episodes=episodes, gamma=gamma, lr=lr, lam=lam, value_lr=value_lr)
