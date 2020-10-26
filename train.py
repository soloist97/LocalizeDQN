import os, json
from collections import deque, Counter

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_value_
from tqdm import tqdm

from model.dqn import DQN
from dataloader import DataLoaderPFG, FastVOCLocalization
from utils.bbox import next_bbox_by_action
from utils.loss import compute_td_loss
from utils.explore import epsilon_by_epoch
from utils.replay import ReplayBuffer
from utils.reward import reward_by_bboxes
from evaluate import evaluate

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_TB = False
CONFIG_PATH = './model_params'
MODEL_NAME = 'debug_ratio'
TOTAL_EPOCH = 10


def set_args():

    args = dict()

    # General settings
    args['model_name'] = MODEL_NAME
    args['voc2007_path'] = './data/voc2007'
    args['feature_map_path'] = {
        'trainval': './data/feature_map_2007_trainval.h5',
        'test': './data/feature_map_2007_test.h5'
    }
    args['img_info_path'] = {
        'trainval': './data/img_info_2007_trainval.pkl',
        'test': './data/img_info_2007_test.pkl'
    }
    args['fm_to_memory'] = True
    args['display_intervals'] = 500
    args['max_size'] = 500  # longest edge

    # Model settings
    args['max_history'] = 3
    args['num_actions'] = (5, 8)

    # Training Settings
    args['total_epochs'] = TOTAL_EPOCH
    args['max_steps'] = 15
    args['replay_capacity'] = 80000  # ~ len(trainval) * 16
    args['replay_initial'] = 3000
    args['target_update'] = 1  # epochs
    args['gamma'] = 0.9
    args['shuffle'] = False  # whether shuffle voc_trainval
    args['epsilon_duration'] = 8

    args['lr'] = 3e-4
    args['batch_size'] = 16
    args['grad_clip'] = 1.

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
    dqn = DQN(num_actions=args['num_actions'], max_history=args['max_history'])

    target_dqn = DQN(num_actions=args['num_actions'], max_history=args['max_history'])
    target_dqn.load_state_dict(dqn.state_dict())

    # move model to GPU before optimizer
    dqn = dqn.to(device)
    target_dqn = target_dqn.to(device)

    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=args['lr'])

    # === prepare data loader ===
    voc_dataset = FastVOCLocalization(args['feature_map_path']['trainval'], args['img_info_path']['trainval'],
                                      args['fm_to_memory'])
    voc_loader = DataLoaderPFG(
                    voc_dataset,
                    batch_size=1, shuffle=args['shuffle'], num_workers=2, pin_memory=True,
                    collate_fn=FastVOCLocalization.collate_fn
                 )

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    # === start ====
    replay_buffer = ReplayBuffer(args['replay_capacity'], voc_dataset)

    for epoch in range(args['total_epochs']):

        epsilon = epsilon_by_epoch(epoch, duration=args['epsilon_duration'])
        # update target network
        if epoch > 0 and epoch % args['target_update'] == 0:
            print('[INFO]: update target dqn')
            target_dqn.load_state_dict(dqn.state_dict())
        if USE_TB:
            writer.add_scalar('training/epsilon', epsilon, epoch)

        for it, (image_idx, img_shape, bbox_gt_list) in tqdm(enumerate(voc_loader), total=len(voc_loader)):

            feature_map = voc_dataset.get_feature_map(image_idx).to(device)
            img_shape = img_shape[0]
            bbox_gt_list = bbox_gt_list[0]
            image_idx = image_idx[0]

            cur_bbox = (0., 0., img_shape[0], img_shape[1])
            history_actions = deque(maxlen=args['max_history'])  # deque of int
            hit_flags = [0] * len(bbox_gt_list)  # use 0 instead of -1 in original paper
            all_rewards = list()
            all_actions = list()

            state = (image_idx, cur_bbox, history_actions)

            for step in range(args['max_steps']):

                # agent
                action = dqn.act((feature_map, *state[1:]), epsilon)

                # environment
                next_bbox = next_bbox_by_action(cur_bbox, action, img_shape)
                history_actions.append(action)

                next_state = (image_idx, next_bbox, history_actions.copy())
                reward, hit_flags = reward_by_bboxes(cur_bbox, next_bbox, bbox_gt_list, hit_flags)

                # replay
                replay_buffer.push(state, action, reward, next_state, step == args['max_steps'] - 1)
                if len(replay_buffer) >= args['replay_initial']:
                    loss = compute_td_loss(dqn, target_dqn, replay_buffer, args['batch_size'], args['gamma'], device)
                    if USE_TB:
                        writer.add_scalar('training/loss', loss.item(),
                                          (epoch * len(voc_loader) + it) * args['max_steps'] + step)

                    optimizer.zero_grad()
                    loss.backward()
                    if args['grad_clip'] > 0:
                        clip_grad_value_(dqn.parameters(), args['grad_clip'])
                    optimizer.step()

                # state transition
                state = next_state
                cur_bbox = next_bbox

                # for display
                all_rewards.append(reward)
                all_actions.append(action)

            if USE_TB:
                writer.add_scalar('training/reward', sum(all_rewards), epoch * len(voc_loader) + it)

            if it % args['display_intervals'] == 0:
                tqdm.write('[{}][{}] \n rewards {}:{} \n actions {}'.format(epoch, it, sum(all_rewards),
                                                                            all_rewards, all_actions))

        save_model(dqn, optimizer, epoch)

        pr_result, _, all_action_pred = evaluate(dqn, 'test', args, device, (0.3, 0.5, 0.7))

        for thr in pr_result.keys():
            print('[IOU threshold]: ', thr)
            print('Precision: {:.4f}   Recall: {:.4f}'.format(pr_result[thr]['P'], pr_result[thr]['R']))

        print('[Action Pairs]: ')
        c = Counter([tuple(ap) for aps in all_action_pred for ap in aps])
        for k, v in c.items():
            print(k, v)

        if USE_TB:
            for thr in pr_result.keys():
                writer.add_scalar('evaluating/Precision-th-{}'.format(thr), pr_result[thr]['P'], epoch)
                writer.add_scalar('evaluating/Recall-th-{}'.format(thr), pr_result[thr]['R'], epoch)

    if USE_TB:
        writer.close()


if __name__ == '__main__':

    training_args = set_args()
    train(training_args)
