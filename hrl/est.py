import numpy as np


def divide_no_nan(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


class Estimator:

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def reset(self):
        pass

    @property
    def action_values(self):
        return []

    def step(self, action, reward):
        pass


class SampleAverage(Estimator):

    def __init__(self, n_actions, initial_value):
        self.initial_value = initial_value
        super().__init__(n_actions)

    def reset(self):
        self._action_rewards = np.full(self.n_actions, self.initial_value)
        self._actions_times = np.zeros(self.n_actions)

    def step(self, action, reward):
        self._actions_times[action] += 1
        self._action_rewards[action] += reward

    @property
    def action_values(self):
        return divide_no_nan(self._action_rewards, self._actions_times)


class ExponentialDecay(Estimator):
    pass