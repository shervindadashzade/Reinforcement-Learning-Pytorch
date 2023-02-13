import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt 


W = 0.6
b = 0.3
learning_rate = 1e-2
x = torch.arange(100).unsqueeze(1).type(torch.float)


y = W * x + b



class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(1,1)
    
    def forward(self,x):
        output = self.linear1(x)
        return output

model = NeuralNet()

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

epochs = 3000

loss_total = []

for epoch in range(epochs):
    y_hat = model(x)
    loss = loss_func(y_hat,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_total.append(loss.data)
    if epoch % 50 == 0:
        print(f'epoch: {epoch}, loss: {loss}')
plt.figure()
plt.plot(y.numpy(), color='green')
plt.plot(y_hat.detach().numpy(), color='red')
plt.figure()
plt.plot(loss_total)
plt.show()