import random
from collections import deque

import torch


class ReplayBuffer(object):

    def __init__(self, capacity):

        self.tensor_buffer = dict()  # {iid : (1, 3, 224, 224)}
        self.buffer = deque(maxlen=capacity)

    def push(self, iid, state, action, reward, next_state):
        """

        :param iid: (int)
        :param state: (tuple) (tensor, tuple, deque)
        :param action: (int)
        :param reward: (float)
        :param next_state: (tuple) (tensor, tuple, deque)
        :return:
        """
        assert state[0].shape[0] == 1, "Not support batch input"

        # save memory
        if iid not in self.tensor_buffer.keys():
            self.tensor_buffer[iid] = state[0].cpu()

        state = (iid, state[1], state[2])
        next_state = (iid, next_state[1], next_state[2])

        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size, device="cpu"):

        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))

        iids, bboxes, hs = zip(*state)
        state = (torch.cat([self.tensor_buffer[iid] for iid in iids], dim=0).to(device), list(bboxes), list(hs))

        _, bboxes, hs = zip(*next_state)
        next_state = (state[0].clone(), list(bboxes), list(hs))

        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)
