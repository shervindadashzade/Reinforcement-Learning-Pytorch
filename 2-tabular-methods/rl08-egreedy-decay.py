import gym
import time
import torch


from importlib_metadata import entry_points

import matplotlib.pyplot as plt
plt.style.use('ggplot')


env = gym.make('FrozenLake-v1', is_slippery=False)



num_episodes = 1000
steps_total = []
reward_total = []
egreedy_total = []
num_of_states = env.observation_space.n
num_of_actions = env.action_space.n

gamma = 0.9
learning_rate = 0.9
egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.999
Q = torch.zeros([num_of_states, num_of_actions])



for episode in range(num_episodes):
    state = env.reset()

    step = 0

    while True:
        step +=1

        if torch.rand(1).item() > egreedy:
            rand_item = Q[state] + torch.randn(1,num_of_actions) / 1000
        
            action = torch.max(rand_item,1)[1][0].item()
        else:
            action = env.action_space.sample()

        if egreedy > egreedy_final:
            egreedy *= egreedy_decay
        
        egreedy_total.append(egreedy)

        new_state, reward, done, info = env.step(action)

        Q[state,action] = (1-learning_rate) * Q[state,action] + learning_rate * (reward + gamma * torch.max(Q[new_state]).item())
        
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
plt.show()
print(f'average steps taken: {sum(steps_total)/num_episodes}')
print(f'percentage of successfully finished episodes: {sum(reward_total) / num_episodes}')
print(f'percentage of successfully finished episodes (last 100 episodes): {sum(reward_total[-100:]) / 100}')
print(f'Q variable : {Q}')
plt.figure(figsize=(12,5))
plt.title('Reward per episodes')
plt.bar(torch.arange(len(reward_total)),reward_total, alpha = 0.6, color='green')
plt.show()

plt.figure(figsize=(12,5))
plt.title('Egreedy per episodes')
plt.bar(torch.arange(len(egreedy_total)),egreedy_total, alpha = 0.6, color='blue')
plt.show()