import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()
done = False

observations = []

class CartPoleAgent:
    def __init__(self):
        self.value = [0, 0]
        self.parameters = np.random.rand(4) * 2 - 1
        self.best_reward = float("-inf")
        self.alpha = 0.1

    def update(self, total_reward, method='hill_climbing'):
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_parameters = self.parameters
            print("Updated params")
        if method == 'random':
            self.parameters = np.random.rand(4) * 2 - 1
        elif method == 'hill_climbing':
            self.update_step = (np.random.rand(4) * 2 - 1)
            self.parameters = self.best_parameters
            self.parameters += self.alpha * self.update_step
    def action(self, observation):
        action = 0 if np.matmul(self.parameters, observation) < 0 else 1
        return action

def run_episode(env, agent):
    observation = env.reset()
    total_reward = 0
    for _ in range(200):
        observation, reward, done, info = env.step(agent.action(observation))
        total_reward += reward
        if done:
            break
    return total_reward

def main():
    agent = CartPoleAgent()
    observation = [0] * 4
    rewards = []
    total_reward = 0
    for _ in range(100):
        env.reset()
        total_reward = run_episode(env, agent)
        agent.update(total_reward)
        rewards.append(total_reward)
    plt.hist(total_reward)
    plt.title("Total rewards for CartPole agent.")


main()
