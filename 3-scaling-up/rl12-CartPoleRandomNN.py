from math import gamma
import gym
import torch
import random
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
import math

env = gym.make('CartPole-v0')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

#### Parameters #####
learning_rate = 0.01
num_episodes = 500
gamma = 0.99

hidden_layer = 128

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 500
######################




def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon




input_features = env.observation_space.shape[0]
output_features = env.action_space.n

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_features, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output_features)
        self.tanh = nn.Tanh()
    
    def forward(self,x):
        output = self.linear1(x)
        output = self.tanh(output)
        output = self.linear2(output)
        return output



class QNetAgent():
    def __init__(self):
        self.nn = NeuralNet()
        self.optimizer = optim.Adam(self.nn.parameters(), lr= learning_rate)
        self.loss_func = nn.MSELoss()
        self.total_loss = []
    def select_action(self, state, epsilon):
        p = torch.rand(1).item()

        if p > epsilon:
            state = torch.Tensor(state)
            net_output = self.nn(state).detach()
            action = torch.max(net_output,dim=0)[1].item()
        else:
            #print('Random...!!!!')
            action = env.action_space.sample()

        return action

    def optimize(self, state,action,new_state,reward,done):
        state = torch.Tensor(state)
        new_state = torch.Tensor(new_state)
        reward = torch.Tensor([reward])

        if done:
            target_value = reward
        else:
            new_state_q = self.nn(new_state).detach()
            target_value = reward + gamma * torch.max(new_state_q)
        
        predicted = self.nn(state)[action]
        
        loss = self.loss_func(predicted, target_value.squeeze(0))
        self.total_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




agent = QNetAgent()




steps_total = []
frame_total = 0
for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    #for step in range(100):
    while True:
        frame_total += 1
        step += 1
        epsilon = calculate_epsilon(frame_total)
        
        action = agent.select_action(state, epsilon)
        
        new_state, reward, done, info = env.step(action)
        
        agent.optimize(state,action,new_state,reward,done)

        state = new_state
        #print(new_state)
        #print(info)
        
        #env.render()
        
        if done:
            steps_total.append(step)
            print("Episode %ith finished after %i steps with epsilon: %.2f" % (i_episode,step,epsilon) )
            break
        

print("Average reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.4f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
#plt.bar(torch.arange(len(steps_total)),steps_total, alpha=0.6,color='green')
plt.show()

env.close()
env.env.close()
