import argparse, json, os
from collections import deque

import torch
from tqdm import tqdm

from model.dqn import DQN
from dataloader import DataLoaderPFG, VOCLocalization
from utils.bbox import next_bbox_by_action, resize_bbox
from utils.metric import bboxPrecisionRecall


def load_config_args(args):

    with open(os.path.join(args['model_root_path'], args['model_name'], 'config.json'), 'r') as f:
        config_args = json.load(f)

    return config_args


def load_model(args):

    # === init model ====
    dqn = DQN(num_actions=tuple(args['num_actions']), max_history=args['max_history'])

    model_checkpoint_path = os.path.join(args['model_root_path'], args['model_name'], args['model_check_point'])

    print('loading checkpoint from {} ...'.format(model_checkpoint_path))
    checkpoint = torch.load(model_checkpoint_path)
    dqn.load_state_dict(checkpoint['dqn'])

    return dqn


@torch.no_grad()
def evaluate(dqn, dataset, args, device, IoU_thresholds=(0.5, 0.6, 0.7)):

    print('[INFO]: start evaluating...')

    # === prepare model ====
    dqn = dqn.to(device)
    dqn.eval()

    # === prepare data loader ====
    voc_loader = DataLoaderPFG(VOCLocalization(args['voc2007_path'], year='2007', image_set=dataset,
                                               download=False, transform=VOCLocalization.get_transform()),
                                        batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                        collate_fn=VOCLocalization.collate_fn)

    # === metric calculator ====
    bbpr = bboxPrecisionRecall(IoU_thresholds)

    # === state evaluating ====

    all_bbox_pred = list()
    all_action_pred = list()
    for img_tensor, original_shape, bbox_gt_list, _ in tqdm(voc_loader, total=len(voc_loader)):

        img_tensor = img_tensor.to(device)
        feature_map = dqn.encoder.encode_image(img_tensor)

        original_shape = original_shape[0]
        bbox_gt_list = bbox_gt_list[0]

        scale_factors = (224. / original_shape[0], 224. / original_shape[1])

        bbox_pred = list()
        action_pred = list()
        states_deque = deque()  # breath first search  (unscaled_bbox, history_actions)
        states_deque.append(((0., 0., original_shape[0], original_shape[1]), deque(maxlen=args['max_steps'])))
        num_nodes_to_add = 2**(args['max_steps'] + 1) - 2

        while(len(states_deque) != 0):

            cur_bbox, history_actions = states_deque.popleft()
            bbox_pred.append(cur_bbox)

            if num_nodes_to_add > 0:

                q_value = dqn((feature_map, resize_bbox(cur_bbox, scale_factors), history_actions))

                # two branch search
                scaling_action = q_value[0, :dqn.num_actions[0]].argmax().item()
                transform_action = dqn.num_actions[0] + q_value[0, -1*dqn.num_actions[1]:].argmax().item()
                action_pred.append((scaling_action, transform_action))

                next_bbox_s = next_bbox_by_action(cur_bbox, scaling_action, original_shape)
                history_actions_s = history_actions.copy()
                history_actions_s.append(scaling_action)

                next_bbox_t = next_bbox_by_action(cur_bbox, transform_action, original_shape)
                history_actions_t = history_actions.copy()
                history_actions_t.append(transform_action)

                states_deque.append((next_bbox_s, history_actions_s))
                states_deque.append((next_bbox_t, history_actions_t))

                num_nodes_to_add -= 2

        all_bbox_pred.append(bbox_pred)
        all_action_pred.append(action_pred)
        bbpr.update(bbox_gt_list, [torch.tensor(b) for b in bbox_pred])

    pr_result = bbpr.evaluate()

    return pr_result, all_bbox_pred, all_action_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root_path", type=str, default="./model_params",
                        help="the root directory of all models")
    parser.add_argument("--dataset", type=str, default="test",
                        help="which dataset to evaluate")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_check_point", type=str, help="model checkpoint path")
    parser.add_argument("--cpu", action='store_true', help="whether to use cpu")
    evaluating_args = vars(parser.parse_args())

    evaluating_args.update(load_config_args(evaluating_args))

    print('==================')
    for k, v in evaluating_args.items():
        print('{}     {}'.format(k, v))
    print('==================')

    device = torch.device('cuda' if torch.cuda.is_available() and not evaluating_args['cpu'] else 'cpu')
    dqn = load_model(evaluating_args)
    pr_result, all_bbox_pred, all_action_pred = evaluate(dqn, evaluating_args['dataset'], evaluating_args, device,
                                                         (0.3, 0.5, 0.7))

    # === display results ===
    for thr in pr_result.keys():
        print('[IOU threshold]: ', thr)
        print('Precision: {:.4f}   Recall: {:.4f}'.format(pr_result[thr]['P'], pr_result[thr]['R']))

    # == save bbox prediction ====
    with open(os.path.join(evaluating_args['model_root_path'],
                           evaluating_args['model_name'], 'results.json'), 'w') as f:
        json.dump({'bbox': all_bbox_pred, 'action': all_action_pred, 'metric': pr_result}, f, indent=2)
