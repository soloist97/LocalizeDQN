from collections import deque


def train(args):

    for epoch in range(args['total_epochs'])

        encoder.train(); dqn.train()
        epsilon = epsilon_by_epoch(epoch)

        for it, (img_tensor, original_shape, bbox_gt_list) in enumerate(voc_trainval_loader):

            original_shape = original_shape[0]
            bbox_gt_list = bbox_gt_list[0]

            cur_bbox = (0., 0., original_shape[0], original_shape[1])
            scale_factors = (224. / original_shape[0], 224. / original_shape[1])
            history_actions = deque(maxlen=args['max_steps'])  # deque of int
            hit_flags = [0] * len(bbox_gt_list)  # use 0 instead of -1 in original paper
            total_reward = 0.

            global_feature, feature_map = encoder(img_tensor)
            bbox_feature = encoder.encode_bbox(feature_map, cur_bbox, scale_factors)
            state = (global_feature, bbox_feature, [history_actions.copy()])

            for step in range(args['max_steps']):

                # agent
                action = dqn.act(state, epsilon)

                # environment
                next_bbox = next_bbox_by_action(cur_bbox, action, original_shape)
                next_bbox_feature = encoder.encode_bbox(feature_map, next_bbox, scale_factors)
                history_actions.append(action)

                next_state = (global_feature, next_bbox_feature, [history_actions.copy()])
                reward, hit_flags = reward_by_bboxes(cur_bbox, next_bbox, bbox_gt_list, hit_flags)

                # replay
                replay_buffer.push(state, action, reward, next_state)
                if len(replay_buffer) > args['batch_size']:
                    loss = compute_td_loss(dqn, replay_buffer, args['batch_size'], args['gamma'])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # for display
                total_reward += reward

            if it % args['display_intervals'] == 0:
                print('[{}][{}] total reward {}'.format(epoch, it, total_reward))
