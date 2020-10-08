import os, json
from collections import deque

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.dqn import DQN
from dataloader import DataLoaderPFG, VOCLocalization
from utils.bbox import ACTION_FUNC_DICT, next_bbox_by_action, resize_bbox
from utils.loss import compute_td_loss
from utils.explore import epsilon_by_epoch
from utils.replay import ReplayBuffer
from utils.reward import reward_by_bboxes

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_TB = False
CONFIG_PATH = './model_params'
MODEL_NAME = 'debug'
TOTAL_EPOCH = 25


def set_args():

    args = dict()

    # General settings
    args['model_name'] = MODEL_NAME
    args['voc2007_path'] = './data/voc2007'
    args['display_intervals'] = 10

    # Model settings
    args['max_history'] = 50
    args['num_inputs'] = 4096 * 2 + len(ACTION_FUNC_DICT) * args['max_history']  # 8842
    args['num_actions'] = (5, 8)
    args['dropout_rate'] = 0.3

    # Training Settings
    args['total_epochs'] = TOTAL_EPOCH
    args['max_steps'] = 50
    args['replay_capacity'] = 250000  # ~ len(trainval) * 50
    args['gamma'] = 0.9
    args['shuffle'] = False  # whether shuffle voc_trainval

    args['lr'] = 1e-3
    args['batch_size'] = 64

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def train(args):

    print('[INFO]: Model {} start training...'.format(MODEL_NAME))

    # === init model ====
    dqn = DQN(num_inputs=args['num_inputs'], num_actions=args['num_actions'],
              max_history=args['max_history'], dropout_rate=args['dropout_rate'])

    # move model to GPU before optimizer
    dqn = dqn.to(device)

    optimizer = torch.optim.Adam(dqn.parameters(), lr=args['lr'])

    # === prepare data loader ====
    voc_trainval_loader = DataLoaderPFG(VOCLocalization(args['voc2007_path'], year='2007', image_set='trainval',
                                                        download=False, transform=VOCLocalization.get_transform()),
                                        batch_size=1, shuffle=args['shuffle'], num_workers=2, pin_memory=True,
                                        collate_fn=VOCLocalization.collate_fn)

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    # === start ====
    replay_buffer = ReplayBuffer(args['replay_capacity'])

    for epoch in range(args['total_epochs']):

        dqn.train()
        epsilon = epsilon_by_epoch(epoch)

        for it, (img_tensor, original_shape, bbox_gt_list) in tqdm(enumerate(voc_trainval_loader),
                                                                   total=len(voc_trainval_loader)):

            img_tensor = img_tensor.to(device)
            original_shape = original_shape[0]
            bbox_gt_list = bbox_gt_list[0]

            cur_bbox = (0., 0., original_shape[0], original_shape[1])
            scale_factors = (224. / original_shape[0], 224. / original_shape[1])
            history_actions = deque(maxlen=args['max_steps'])  # deque of int
            hit_flags = [0] * len(bbox_gt_list)  # use 0 instead of -1 in original paper
            all_rewards = list()

            state = (img_tensor, resize_bbox(cur_bbox, scale_factors), history_actions.copy())

            for step in range(args['max_steps']):

                # agent
                action = dqn.act(state, epsilon)

                # environment
                next_bbox = next_bbox_by_action(cur_bbox, action, original_shape)
                history_actions.append(action)

                next_state = (img_tensor, resize_bbox(next_bbox, scale_factors), history_actions.copy())
                reward, hit_flags = reward_by_bboxes(cur_bbox, next_bbox, bbox_gt_list, hit_flags)

                # replay
                replay_buffer.push(state, action, reward, next_state)
                if len(replay_buffer) > args['batch_size']:
                    loss = compute_td_loss(dqn, replay_buffer, args['batch_size'], args['gamma'], device)
                    if USE_TB:
                        writer.add_scalar('training/loss', loss.item(),
                                          (epoch * len(voc_trainval_loader) + it) * args['max_steps'] + step)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # for display
                all_rewards.append(reward)

            if USE_TB:
                writer.add_scalar('training/reward', sum(all_rewards), epoch * len(voc_trainval_loader) + it)

            if it % args['display_intervals'] == 0:
                print('[{}][{}] rewards {}:{}'.format(epoch, it, sum(all_rewards), all_rewards))

    if USE_TB:
        writer.close()

if __name__ == '__main__':

    training_args = set_args()
    train(training_args)
