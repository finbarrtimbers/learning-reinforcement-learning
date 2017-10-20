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

    def policy_gradient(self, learning_rate=0.01):
        params = tf.get_variable("policy_parameters", [4, 2])
        state = tf.placeholder("float", [None, 4])
        action = tf.placeholder("float", [None, 2])
        linear = tf.matmul(state, params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),
                                           reduction_indices=[1])
        log_probabilities = tf.log(good_probabilities)
        loss = -tf.reduce_sum(log_probabilities)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def value_gradient(self, learning_rate):
        state = tf.placeholder("float", [None, 4])
        w1 = tf.get_variable("w1", [4, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [10, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1, w2) + b2
        newvals = tf.placeholder("float", [None, 1])
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
    pl_probabilities, pl_state = agent.policy_gradient()
    observation = env.reset()
    actions = []
    transitions = []
    total_reward = 0

    # Run episodes
    for _ in range(NUMB_EPISODES):
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_probabilities, feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0, 1) < probs[0][0] else 1

        # record data
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)

        # take action

        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        total_reward += reward
        env.reset()
        total_reward = run_episode(env, agent)

        if done:
            break

    vl_calculated, vl_state, vl_newvals, vl_optimizer = value_gradient()
    update_vals = []
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for future_index in range(future_transitions):
            future_reward += transitions[(future_index) + index][2] * decrease
            decrease *= 0.97
        update_vals.append(future_reward)
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states,
                                      vl_newvals: update_vals_vector})

    create_histogram(rewards,
                     f"Total rewards for CartPole agent with {NUMB_EPISODES}")

main()
