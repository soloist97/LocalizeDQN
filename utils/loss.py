import torch
import torch.nn.functional as F


def compute_td_loss(dqn, replay_buffer, batch_size, gamma=0.9, device='cpu'):
    """

    :param dqn: (model.dqn.DQN)
    :param replay_buffer: (utils.replay.ReplayBuffer)
    :param batch_size: (int)
    :param gamma: (float)
    :param device: (str or torch.device)
    :return: loss (tensor)
    """

    state, action, reward, next_state = replay_buffer.sample(batch_size, device)

    global_feature, feature_map = dqn.encoder(state[0])

    q_values = dqn(state, global_feature, feature_map)  # (batch_size, num_actions)
    next_q_values = dqn(next_state, global_feature, feature_map).detach()  # no gradient along this path

    q_value = q_values[torch.arange(batch_size), action]  # (batch_size, )
    next_q_value = next_q_values.max(dim=-1)[0]
    expected_q_value = reward + gamma * next_q_value

    return F.mse_loss(q_value, expected_q_value)
