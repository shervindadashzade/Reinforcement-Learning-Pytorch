import gym
import time
import torch


from importlib_metadata import entry_points

import matplotlib.pyplot as plt
plt.style.use('ggplot')


env = gym.make('FrozenLake-v1', is_slippery=True)



num_episodes = 1000
steps_total = []
reward_total = []
num_of_states = env.observation_space.n
num_of_actions = env.action_space.n
gamma = 1

Q = torch.zeros([num_of_states, num_of_actions])



for episode in range(num_episodes):
    state = env.reset()

    step = 0

    while True:
        step +=1


        rand_item = Q[state] + torch.randn(1,num_of_actions) / 1000
        #action = env.action_space.sample()
        
        action = torch.max(rand_item,1)[1][0].item()

        new_state, reward, done, info = env.step(action)

        Q[state,action] = reward + gamma * torch.max(Q[new_state]).item()
        
        state = new_state
        #env.render()
        if done:
            steps_total.append(step)
            reward_total.append(reward)
            print(f'episode finished with {step} steps.')
            break
plt.figure(figsize=(12,5))
plt.title('Steps per episodes')
plt.bar(torch.arange(len(steps_total)),steps_total, alpha = 0.6, color='red')
print(f'average steps taken: {sum(steps_total)/num_episodes}')
print(f'percentage of successfully finished episodes: {sum(reward_total) / num_episodes}')
print(f'percentage of successfully finished episodes (last 100 episodes): {sum(reward_total[-100:]) / 100}')
plt.figure(figsize=(12,5))
plt.title('Reward per episodes')
plt.show()
plt.bar(torch.arange(len(reward_total)),reward_total, alpha = 0.6, color='green')
plt.show()