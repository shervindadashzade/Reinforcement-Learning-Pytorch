import gym
import torch
import matplotlib.pyplot as plt
env = gym.make('Taxi-v3')

states_num = env.observation_space.n
actions_num = env.action_space.n
episodes_num = 2000
Q = torch.zeros([states_num, actions_num])
gamma = 0.9
egreedy = 0.9
egreedy_final = 0.01
egreedy_deacy = 0.99


total_steps =[]
total_rewards = []
total_egreedies = []
for episode in range(episodes_num):
    
    state = env.reset()
    step = 0
    temp_reward = 0
    while True:
        step +=1
        rand_values = Q[state] + torch.rand(1,actions_num) / 1000
        action = torch.max(rand_values, dim=1)[1].item()
        
        new_state, reward, done, info = env.step(action)
        Q[state,action] = reward + gamma * torch.max(Q[new_state]).item()
        temp_reward += reward

        state = new_state
        #env.render()

        if done:
            print(f'Episode {episode}th has finished with {step} steps.')
            print(f'The gained reawrd is: {temp_reward}')
            total_steps.append(step)
            total_rewards.append(temp_reward)
            break

print(Q)
plt.figure(figsize=(10,3))
plt.title('Steps')
plt.bar(range(episodes_num), total_steps, alpha = 0.7, color="red")
plt.figure(figsize=(10,3))
plt.title('Rewards')
plt.bar(range(episodes_num), total_rewards, alpha = 0.7, color="green")
plt.show()




