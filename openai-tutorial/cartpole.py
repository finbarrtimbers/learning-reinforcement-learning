import gym
import numpy as np
import pylab
import random

env = gym.make('CartPole-v0')
env.reset()
done = False

observations = []

class CartPoleAgent:
    def __init__(self):
        self.value = [0, 0]
        self.parameters = np.random.rand(4) * 2 - 1
        self.best_reward = float("-inf")

    def update(self, total_reward):
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_parameters = self.parameters
            print("Updated params")
        self.parameters = np.random.rand(4) * 2 - 1

    def action(self, observation):
        action = 0 if np.matmul(self.parameters, observation) < 0 else 1
        return action

agent = CartPoleAgent()
observation = [0] * 4
total_reward = 0
for _ in range(1000):
    env.reset()
    observation = [0] * 4
    total_reward = 0
    while not done:
        #if done:
        #    env.reset()
        #env.render()
        # observation = x, x_dot, theta, theta_dot
        observation, reward, done, info = env.step(agent.action(observation))
        observations.append(observation)
        total_reward += reward
    agent.update(total_reward)

agent.parameters = agent.best_parameters
print(agent.best_reward)
env.reset()
done = False
while not done:
    #if done:
    #    env.reset()
    env.render()
    # observation = x, x_dot, theta, theta_dot
    observation, reward, done, info = env.step(agent.action(observation))
    observations.append(observation)
    total_reward += reward
print(f"total_reward: {total_reward}")

t = range(len(observations))
observations = np.array(observations)
desc = {0: 'x',
        1: 'x_dot',
        2: 'theta',
        3: 'theta_dot'}
for i in range(observations.shape[1]):
    pylab.plot(t, observations[:, i], label=desc[i])
pylab.legend(loc='upper right')
pylab.draw()
pylab.pause(1)
input("Hit enter to close")
pylab.close()
