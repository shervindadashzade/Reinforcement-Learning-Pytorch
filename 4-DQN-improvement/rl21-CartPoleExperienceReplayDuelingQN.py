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
update_target_frequency = 100
learning_rate = 0.01
num_episodes = 200
gamma = 1

hidden_layer = 128
maximum_step = 200
batch_size = 32
capacity = 50000
egreedy = 0.9
egreedy_final = 0
egreedy_decay = 500

double_qn = True
clip_error = False
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
        self.advantage = nn.Linear(hidden_layer, output_features)
        self.value = nn.Linear(hidden_layer, 1)
        self.tanh = nn.Tanh()
    
    def forward(self,x):
        output = self.linear1(x)
        output = self.tanh(output)
        
        out_advantage = self.advantage(output)
        out_value = self.value(output)

        final_output = out_value + out_advantage - out_advantage.mean()
        return final_output


class ExperieceReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state,action,new_state,reward,done):
        transition = (state, action, new_state, reward, done)
        if self.capacity > len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return zip(*random.sample(self.memory, batch_size))
    
    def __len__(self):
        return len(self.memory)


class QNetAgent():
    def __init__(self):
        self.nn = NeuralNet()
        self.target = NeuralNet()
        self.optimizer = optim.Adam(self.nn.parameters(), lr= learning_rate)
        self.loss_func = nn.MSELoss()
        self.total_loss = []
        self.update_counter = 0
    def select_action(self, state, epsilon):
        p = torch.rand(1).item()

        if p > epsilon:
            state = torch.Tensor(state)
            net_output = self.nn(state).detach()
            action = torch.max(net_output,dim=0)[1].item()
        else:
            action = env.action_space.sample()

        return action

    def optimize(self):
        if len(memory) < batch_size:
            return
        
        state, action, new_state, reward, done = memory.sample()
        state = torch.Tensor(state)
        action = torch.Tensor(action).type(torch.int64)
        new_state = torch.Tensor(new_state)
        reward = torch.Tensor(reward)
        done = torch.Tensor(done)
        


        if double_qn:
            best_actions = torch.max(self.nn(new_state).detach(),1)[1]
            max_values = self.target(new_state).gather(1,best_actions.unsqueeze(1)).squeeze(1)
            
            target_value = reward + (1-done) * gamma * max_values
            
        else:
            new_state_q = self.target(new_state).detach()
            max_values = torch.max(new_state_q,1)[0]
            target_value = reward + (1 - done) * gamma * max_values
            #print(target_value.shape)

        nn_output = self.nn(state)
        predicted = nn_output.gather(1,action.unsqueeze(1))
        
        loss = self.loss_func(predicted.squeeze(1), target_value.squeeze(0))
        self.total_loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        
        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp(-1,1)

        self.optimizer.step()
        if self.update_counter % update_target_frequency == 0 :
            self.target.load_state_dict(self.nn.state_dict())
        self.update_counter += 1




agent = QNetAgent()
memory = ExperieceReplay(capacity)



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
        
        memory.push(state, action, new_state, reward, done)

        agent.optimize()

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
print(f"Environment solved after {steps_total.index(maximum_step)}")

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
#plt.bar(torch.arange(len(steps_total)),steps_total, alpha=0.6,color='green')
plt.show()

env.close()
env.env.close()
