import torch
import torch.nn.functional as F


def compute_td_loss(dqn, target_dqn, replay_buffer, batch_size, gamma=0.9, device='cpu'):
    """

    :param dqn: (model.dqn.DQN)
    :param target_dqn: (model.dqn.DQN)
    :param replay_buffer: (utils.replay.ReplayBuffer)
    :param batch_size: (int)
    :param gamma: (float)
    :param device: (str or torch.device)
    :return: loss (tensor)
    """

    dqn.train()
    target_dqn.eval()

    state, action, reward, next_state, done = replay_buffer.sample(batch_size, device)

    q_values = dqn(state)  # (batch_size, num_actions)
    with torch.no_grad():
        next_q_values = target_dqn(next_state).detach()  # no gradient along this path

    q_value = q_values[torch.arange(batch_size), action]  # (batch_size, )
    next_q_value = next_q_values.max(dim=-1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    return F.smooth_l1_loss(q_value, expected_q_value), q_value, expected_q_value
