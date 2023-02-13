import gym
import matplotlib.pyplot as plt

print(f'Gym version is: {gym.__version__}')

path2videos = './videosCartPole/'
env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, path2videos, video_callable = lambda episode: True, force=True)

#env = gym.wrappers.RecordVideo(env, path2videos, episode_trigger= lambda x: True)

num_episodes = 1000




steps = []

for episode in range(num_episodes):
    state = env.reset()
    step = 0
    while True:
        step +=1
        action = env.action_space.sample()
        new_state, reward, done, info  = env.step(action)
        if done:
            steps.append(step)
            print(f'episode finished after {step} steps')
            break

env.close()