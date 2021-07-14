import random
from collections import deque

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, item):
        """Save a transition"""
        self.buffer.append(item)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
