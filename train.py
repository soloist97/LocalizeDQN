import os, json
from collections import deque

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_value_
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
TOTAL_EPOCH = 15


def set_args():

    args = dict()

    # General settings
    args['model_name'] = MODEL_NAME
    args['voc2007_path'] = './data/voc2007'
    args['display_intervals'] = 10

    # Model settings
    args['max_history'] = 15
    args['num_inputs'] = 4096 * 2 + len(ACTION_FUNC_DICT) * args['max_history']  # 8842
    args['num_actions'] = (5, 8)
    args['dropout_rate'] = 0.3

    # Training Settings
    args['total_epochs'] = TOTAL_EPOCH
    args['max_steps'] = 15
    args['replay_capacity'] = 80000  # ~ len(trainval) * 15
    args['gamma'] = 0.9
    args['shuffle'] = False  # whether shuffle voc_trainval
    args['epsilon_duration'] = int(TOTAL_EPOCH * 2 / 5)

    args['lr'] = 1e-3
    args['batch_size'] = 16
    args['grad_clip'] = 5.

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def save_model(dqn, optimizer, epoch, checkpoint_name=None):

    states = {'config_path': os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'),
              'dqn': dqn.state_dict(),
              'optimizer': optimizer.state_dict(),
              'epoch': epoch}

    filename = os.path.join('model_params', MODEL_NAME,
                            '{}.pth.tar'.format(checkpoint_name if checkpoint_name else 'epoch_'+str(epoch)))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(states, filename)


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
                                        batch_size=1, shuffle=args['shuffle'], num_workers=1, pin_memory=True,
                                        collate_fn=VOCLocalization.collate_fn)

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    # === start ====
    replay_buffer = ReplayBuffer(args['replay_capacity'])

    for epoch in range(args['total_epochs']):

        dqn.train()
        epsilon = epsilon_by_epoch(epoch, duration=args['epsilon_duration'])

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
                    if args['grad_clip'] > 0:
                        clip_grad_value_(dqn.parameters(), args['grad_clip'])
                    optimizer.step()

                # for display
                all_rewards.append(reward)

            if USE_TB:
                writer.add_scalar('training/reward', sum(all_rewards), epoch * len(voc_trainval_loader) + it)

            if it % args['display_intervals'] == 0:
                tqdm.write('[{}][{}] rewards {}:{}'.format(epoch, it, sum(all_rewards), all_rewards))

        save_model(dqn, optimizer, epoch)

    if USE_TB:
        writer.close()


if __name__ == '__main__':

    training_args = set_args()
    train(training_args)
