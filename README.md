

# Part 1: Vanilla Policy Gradient

- Agent is trained with a single policy gradient update on the policy network.
- During policy update, a simple baseline is calculated as the average of the last 100 discounted returns.
- To encourage exploration, the action variance is shifted up by some value, starting from 1.0 and exponentially decaying to 0.05 over 1 million episodes.

## Reward Plot

![reward_v](https://github.com/user-attachments/assets/b1b031bd-b14d-4c4f-a53e-796bcee3466e)

## Example Videos

<img src="https://github.com/user-attachments/assets/9465eb0a-9e11-42c3-9876-a13261fa165b" width="300"/>
<img src="https://github.com/user-attachments/assets/ffb8be07-249a-459f-b2f1-9dd5126df5f3" width="300"/>
<img src="https://github.com/user-attachments/assets/e4e49a0b-5af8-4035-9df5-4c2ff978f2a5" width="300"/>

# Part 2: Soft Actor-Critic

- Agent is trained with a policy and a value network.
- For advantage estimation, Generalized Advantage Estimation (GAE) is used. For every step in episode, GAE estimates the advantage as an exponential average of all the possible temporal differences, weighing 1-step TD the most. 
- To encourage exploration, the action variance is shifted up by some value, starting from 0.5 and exponentially decaying to 0.1 over 50K episodes.

## Reward Plot

![reward_ac](https://github.com/user-attachments/assets/aa8024d7-f40f-4ff1-8915-d71ecb258306)

## Example Videos

<img src="https://github.com/user-attachments/assets/32dce771-fd80-41ec-a58f-2b524d7577e8" width="300"/>
<img src="https://github.com/user-attachments/assets/0f989f14-7f40-4ca0-9abe-a948f5bebe9c" width="300"/>
<img src="https://github.com/user-attachments/assets/b821facc-1d52-4287-a7d4-f214aae78899" width="300"/>

Environment: [Gymnasium Pusher Environment](https://gymnasium.farama.org/environments/mujoco/pusher/)
