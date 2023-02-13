import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
steps = []
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    step = 0
    while True:
        step += 1 
        action = env.action_space.sample()
        new_state, reward, done, _, info = env.step(action)
        #env.render()
        if done:
            steps.append(step)
            print(f'episode finished after {step} steps')
            break
plt.plot(steps)
plt.show()
env.close()