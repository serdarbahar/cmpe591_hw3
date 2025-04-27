Environment: [Gymnasium Pusher Environment](https://gymnasium.farama.org/environments/mujoco/pusher/)

# Part 1: Vanilla Policy Gradient

- Agent is trained with a single policy gradient update on the policy network.
- During policy update, a simple baseline is calculated as the average of the last 100 discounted returns.
- To encourage exploration, the action variance is shifted up by some value, starting from 1.0 and exponentially decaying to 0.05 over 1 million episodes.

## Reward Plot

![reward_v](https://github.com/user-attachments/assets/b1b031bd-b14d-4c4f-a53e-796bcee3466e)

## Example Videos

![episode_2](https://github.com/user-attachments/assets/283b04c6-4c81-4181-bed3-8775f97f0e7a)


# Part 2: Soft Actor-Critic

- Agent is trained with a policy and a value network.
- For advantage estimation, Generalized Advantage Estimation (GAE) is used. For every step in episode, GAE estimates the advantage as an exponential average of all the possible temporal differences, weighing 1-step TD the most. 
- To encourage exploration, the action variance is shifted up by some value, starting from 0.5 and exponentially decaying to 0.1 over 50K episodes.

## Reward Plot

![reward_ac](https://github.com/user-attachments/assets/aa8024d7-f40f-4ff1-8915-d71ecb258306)

## Example Videos


