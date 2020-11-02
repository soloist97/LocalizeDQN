import os, json
from collections import deque, Counter

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_value_
from tqdm import tqdm

from model.dqn import DQN
from dataloader import DataLoaderPFG, VOCLocalization, CombinedDataset
from utils.bbox import next_bbox_by_action, resize_bbox
from utils.loss import compute_td_loss
from utils.explore import epsilon_by_epoch
from utils.replay import ReplayBuffer
from utils.reward import reward_by_bboxes, init_hit_flags
from evaluate import evaluate

torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_TB = False
CONFIG_PATH = './model_params'
MODEL_NAME = 'debug'
TOTAL_EPOCH = 10


def set_args():

    args = dict()

    # General settings
    args['model_name'] = MODEL_NAME
    args['voc2007_path'] = './data/voc2007'
    args['voc2012_path'] = './data/voc2012'
    args['display_intervals'] = 500

    # Model settings
    args['max_history'] = 3
    args['num_actions'] = (5, 8)

    # Training Settings
    args['total_epochs'] = TOTAL_EPOCH
    args['max_steps'] = 15
    args['replay_capacity'] = 17000 * 15  # ~ len(trainval) * 15
    args['replay_initial'] = 1000 * 15
    args['target_update'] = 4000  # pics
    args['evaluate_duration'] = 16000  # pics
    args['gamma'] = 0.9
    args['shuffle'] = False  # whether shuffle voc_trainval
    args['epsilon_duration'] = 8

    args['lr'] = 3e-4
    args['weight_decay'] = 1e-4
    args['batch_size'] = 16
    args['grad_clip'] = 1.

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def save_model(dqn, optimizer, epoch, it, checkpoint_name=None):

    states = {'config_path': os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'),
              'dqn': dqn.state_dict(),
              'optimizer': optimizer.state_dict(),
              'epoch': epoch,
              'it': it}

    filename = os.path.join('model_params', MODEL_NAME,
                            '{}.pth.tar'.format(checkpoint_name if checkpoint_name else 'epoch_{}_iter_{}'.format(epoch, it)))
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

    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # === prepare data loader ===
    voc_loader = DataLoaderPFG(CombinedDataset(
                                    VOCLocalization(args['voc2007_path'], year='2007', image_set='trainval',
                                                   download=False, transform=VOCLocalization.get_transform()),
                                    VOCLocalization(args['voc2012_path'], year='2012', image_set='trainval',
                                                    download=False, transform=VOCLocalization.get_transform())),
                                batch_size=1, shuffle=args['shuffle'], num_workers=1, pin_memory=True,
                                collate_fn=VOCLocalization.collate_fn)

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    # === start ====
    replay_buffer = ReplayBuffer(args['replay_capacity'])

    for epoch in range(args['total_epochs']):

        epsilon = epsilon_by_epoch(epoch, duration=args['epsilon_duration'])
        if USE_TB:
            writer.add_scalar('training/epsilon', epsilon, epoch)

        for it, (img_tensor, original_shape, bbox_gt_list, image_idx) in tqdm(enumerate(voc_loader),
                                                                              total=len(voc_loader)):

            img_tensor = img_tensor.to(device)
            original_shape = original_shape[0]
            bbox_gt_list = bbox_gt_list[0]
            image_idx = image_idx[0]

            cur_bbox = (0., 0., original_shape[0], original_shape[1])
            scale_factors = (224. / original_shape[0], 224. / original_shape[1])
            history_actions = deque(maxlen=args['max_history'])  # deque of int
            hit_flags = init_hit_flags(cur_bbox, bbox_gt_list)  # use 0 instead of -1 in original paper
            all_rewards = list()
            all_actions = list()

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
                replay_buffer.push(image_idx, state, action, reward, next_state, step == args['max_steps'] - 1)
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

            # update target network
            if len(replay_buffer) >= args['replay_initial'] and \
                    (epoch * len(voc_loader) + it) % args['target_update'] == 0:

                # evaluate before update
                if (epoch * len(voc_loader) + it) % args['evaluate_duration'] == 0:

                    save_model(dqn, optimizer, epoch, it)

                    pr_result, _, all_action_pred = evaluate(dqn, 'test', args, device, (0.3, 0.5, 0.7))

                    for thr in pr_result.keys():
                        print('[IOU threshold]: ', thr)
                        print('Precision: {:.4f}   Recall: {:.4f}'.format(pr_result[thr]['P'], pr_result[thr]['R']))

                    print('[Action Pairs]: ')
                    c = Counter([tuple(ap) for aps in all_action_pred for ap in aps])
                    for k, v in c.items():
                        print(k, v)

                    #FIXME change epoch to iter
                    if USE_TB:
                        for thr in pr_result.keys():
                            writer.add_scalar('evaluating/Precision-th-{}'.format(thr), pr_result[thr]['P'], epoch)
                            writer.add_scalar('evaluating/Recall-th-{}'.format(thr), pr_result[thr]['R'], epoch)

                # update target network
                tqdm.write('[INFO]: update target dqn')
                target_dqn.load_state_dict(dqn.state_dict())

            if USE_TB:
                writer.add_scalar('training/reward', sum(all_rewards), epoch * len(voc_loader) + it)

            if it % args['display_intervals'] == 0:
                tqdm.write('[{}][{}] \n rewards {}:{} \n actions {}'.format(epoch, it, sum(all_rewards),
                                                                            all_rewards, all_actions))
    save_model(dqn, optimizer, args['total_epochs'], 0)

    if USE_TB:
        writer.close()


if __name__ == '__main__':

    training_args = set_args()
    train(training_args)
