import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.8
        self.count = 0

    def get_probs(self, state):
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1 - self.epsilon + (self.epsilon / self.nA)
        # if np.random.random() > eps:  # select greedy action with probability epsilon\n",
        #     return np.argmax(self.Q[state])
        # else:                     # otherwise, select an action randomly\n",
        #     return np.random.choice(np.arange(env.action_space.n))"
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state in self.Q:
            action = np.random.choice(np.arange(self.nA), p=self.get_probs(state))
        else:
            action = np.random.choice(np.arange(self.nA))

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        gamma = 1.0

        # update Q
        if next_state in self.Q:
            next_expected_return = np.sum(self.Q[next_state] * self.get_probs(next_state))
        else:
            next_expected_return = 0

        # alpha = 0.1 + (0.4 * (20000 - self.count) / 20000)
        # print(alpha)
        alpha = 0.3
        # if done:
        #     self.Q[state][action] = 20
        # else:
        self.Q[state][action] += alpha * (reward + gamma * next_expected_return - self.Q[state][action])

        if self.epsilon > 0.00001:
            self.epsilon -= 0.00001
        # prepare for next loop\n",
        # state = next_state
