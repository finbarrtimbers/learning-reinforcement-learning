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

    def policy_gradient():
        params = tf.get_variable("policy_parameters", [4, 2])
        state

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

def create_histogram(values, title, legend=False):
    plt.hist(values)
    plt.title("Total rewards for CartPole agent.")
    if legend:
        plt.legend(loc='uper right')
    plt.draw()
    plt.pause(1)
    input("<Hit enter to close>")
    plt.close()

def main():
    # Config
    NUMB_EPISODES = 1000

    # Initialize values
    agent = CartPoleAgent()
    observation = [0] * 4
    rewards = []
    total_reward = 0

    # Run episodes
    for _ in range(NUMB_EPISODES):
        env.reset()
        total_reward = run_episode(env, agent)
        agent.update(total_reward)
        rewards.append(total_reward)
    create_histogram(rewards,
                     f"Total rewards for CartPole agent with {NUMB_EPISODES}")

main()
