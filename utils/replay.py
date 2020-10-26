import random
from collections import deque

import torch


class ReplayBuffer(object):

    def __init__(self, capacity, dataset):

        self.dataset = dataset
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """

        :param state: (tuple) (index, tuple, deque)
        :param action: (int)
        :param reward: (float)
        :param next_state: (tuple) (index, tuple, deque)
        :param done: (bool)
        :return:
        """
        assert isinstance(state[0], int), "Not support batch input"

        self.buffer.append((state, action, reward, next_state, 1 if done else 0))

    def sample(self, batch_size, device="cpu"):

        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        iids, bboxes, hs = zip(*state)
        state = (self.dataset.get_feature_map(iids).to(device), list(bboxes), list(hs))

        _, bboxes, hs = zip(*next_state)
        next_state = (state[0].clone(), list(bboxes), list(hs))

        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
