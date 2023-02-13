# Reinforcement Learning with PyTorch

This repository contains my implementations based on the Udemy Reinforcement Learning with PyTorch course by Atamai AI Team. The project structure is described in the following:

### 1 - Introduction:
Set up and created CartPole and FrozenLake environments using OpenAI Gym and explored the environment using random sampling from environment action space.

### 2 - Tabular Methods
Solved the FrozenLake in both the deterministic and stochastic environment using Bellman Equation and Q learning, also introduced the e-greedy algorithm to explore the environment by random chance.

### 3 - Scaling Up
Solved the CartPole environment using Deep Q learning and implemented a simple FC neural network to learn our Q function.

### 4- Deep Q Neural Network Improvement
Implemented the Experience Replay, Double Deep Q learning, and Dueling Double Deep Q learning mechanism introduced by DeepMind to help our Neural Network to learn the Q function more accurately.

### 5- Deep Q Neural Network with video output
Used CNNs network to extract environment features from a Grayscale image of the Pong Game environment alongside Deep Q learning for our agent to learn to play Pong Game and dominate the opponent.

Here is a video of our agent(Green paddle) dominating the game:


![last_episode](https://user-images.githubusercontent.com/42402986/218476792-698dd8eb-5a84-4315-af67-f46c50fd2347.gif)


Let's take a look at how our agent has learned the game through different episodes:
![Pong-results](https://user-images.githubusercontent.com/42402986/218475670-f85de7fb-f302-4c61-9a1b-29fee9d3c2ab.png)

Moreover recorded videos and trained model weights are available in the: 5-DQN-with-video-output/results directory.
