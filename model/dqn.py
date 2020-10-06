import random

import torch
from torch import nn


class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions=(5, 8), max_history=50, dropout_rate=0.3):

        super(DQN, self).__init__()

        assert isinstance(num_actions, tuple)

        self.num_inputs = num_inputs
        self.num_actions = num_actions  # (scaling_action, local_translation_action)
        self.max_history = max_history
        self.dropout_rate = dropout_rate

        # delete the last 1024 Linear layer for simplicity
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_actions[0] + num_actions[1])
        )

    def forward(self, state):
        """

        :param state: (tuple) (tensor, tensor, list(deque))
        :return: (tensor) (batch_size, num_actions)
        """

        global_feature, bbox_feature, history_actions = state
        batch_size = global_feature.shape[0]
        device = global_feature.device

        # (batch_size, max_history, total_num_actions)
        history_emb= torch.zeros(batch_size, self.max_history, (self.num_actions[0] + self.num_actions[1]))
        for i in range(batch_size):
            if len(history_actions[i]) == 0:
                continue
            history_emb[i, torch.arange(self.max_history), torch.tensor(history_actions[i], dtype=torch.long)] = 1.
        history_emb = history_emb.to(device)

        # (batch_size, num_inputs)
        x = torch.cat([global_feature, bbox_feature, history_emb.view(batch_size, -1)], dim=-1)

        return self.layers(x)  # (batch_size, total_num_actions)

    @torch.no_grad()
    def act(self, state, epsilon):

        assert len(state[2]) == 1, "Do not support batch input"

        # record mode
        is_training = self.training
        if is_training:
            self.eval()

        if random.random() > epsilon:
            q_value = self.forward(state)  # (1, num_actions)
            if random.random() > 0.5:
                action  = q_value[0, :self.num_actions[0]].argmax().item()
            else:
                action = q_value[0, -1*self.num_actions[1]:].argmax().item()
        else:
            action = random.randrange(self.num_actions[0] + self.num_actions[1])

        # set mode back
        if is_training:
            self.train()

        return action
