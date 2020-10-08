import random
from collections import deque

import torch


class ReplayBuffer(object):

    def __init__(self, capacity):

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """

        :param state: (tuple) (tensor, tensor, list(deque))
        :param action: (int)
        :param reward: (float)
        :param next_state: (tuple) (tensor, tensor, list(deque))
        :return:
        """

        # save memory for GPU
        state = tuple(s.cpu() if isinstance(s, torch.Tensor) else s for s in state)
        next_state = tuple(s.cpu() if isinstance(s, torch.Tensor) else s for s in next_state)

        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size, device="cpu"):

        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))

        gf, fm, hs = zip(*state)
        state = (torch.cat(gf, dim=0).to(device), torch.cat(fm, dim=0).to(device), [h[0] for h in hs])

        gf, fm, hs = zip(*next_state)
        next_state = (torch.cat(gf, dim=0).to(device), torch.cat(fm, dim=0).to(device), [h[0] for h in hs])

        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)
