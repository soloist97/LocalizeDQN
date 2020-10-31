import random
from collections import deque

import torch
from torch import nn

from model.encoder import RESNET50Encoder


class DQN(nn.Module):

    def __init__(self, num_actions=(5, 8), max_history=50):

        super(DQN, self).__init__()

        assert isinstance(num_actions, tuple)

        self.num_actions = num_actions  # (scaling_action, local_translation_action)
        self.max_history = max_history

        self.encoder = RESNET50Encoder()

        # delete few Linear layers for simplicity
        self.num_inputs = self.encoder.num_outputs * 2 + (num_actions[0] + num_actions[1]) * max_history
        self.layers = nn.Sequential(
            nn.Linear(self.num_inputs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions[0] + num_actions[1])
        )

    def forward(self, state):
        """

        :param state: (tuple) (img_tensor or feature map, list(bbox), list(deque))
        :return: (tensor) (batch_size, num_actions)
        """

        img_input, cur_bbox, history_actions = state
        if isinstance(cur_bbox, tuple):
            cur_bbox = [cur_bbox]
        if isinstance(history_actions, deque):
            history_actions = [history_actions]

        batch_size = img_input.shape[0]
        device = img_input.device

        global_feature, bbox_feature = self.encoder(img_input, cur_bbox)

        # (batch_size, max_history, total_num_actions)
        history_emb= torch.zeros(batch_size, self.max_history, (self.num_actions[0] + self.num_actions[1]))
        for i in range(batch_size):
            if len(history_actions[i]) == 0:
                continue
            history_emb[i, torch.arange(len(history_actions[i])),
                        torch.tensor(history_actions[i], dtype=torch.long)] = 1.
        history_emb = history_emb.to(device)

        # (batch_size, num_inputs)
        x = torch.cat([global_feature, bbox_feature, history_emb.view(batch_size, -1)], dim=-1)

        return self.layers(x)  # (batch_size, total_num_actions)

    @torch.no_grad()
    def act(self, state, epsilon):

        # record mode
        is_training = self.training
        if is_training:
            self.eval()

        if random.random() > epsilon:
            q_value = self.forward(state)  # (1, num_actions)
            assert q_value.shape[0] == 1, "Not support batch input"
            if random.random() > 0.5:
                action  = q_value[0, :self.num_actions[0]].argmax().item()
            else:
                action = self.num_actions[0] + q_value[0, -1*self.num_actions[1]:].argmax().item()
        else:
            action = random.randrange(self.num_actions[0] + self.num_actions[1])

        # set mode back
        if is_training:
            self.train()

        return action
