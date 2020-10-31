import  torch
from torchvision.ops.boxes import box_iou


def init_hit_flags(start_bbox, bbox_gt_list):
    """

    :param start_bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param bbox_gt_list: (list of tensor) [(4,)]
    :return: hit_flags: (list)
    """

    assert len(bbox_gt_list) > 0, "invalid bbox_gt_list input"

    # (1, 4)
    bbox = torch.tensor([start_bbox], dtype=bbox_gt_list[0].dtype).to(bbox_gt_list[0].device)

    # (M, 4)
    bbox_gt = torch.stack(bbox_gt_list, dim=0)

    bbox_iou = box_iou(bbox, bbox_gt)  # (1, M)

    return (bbox_iou[0] > 0.5).long().tolist()


def reward_by_bboxes(cur_bbox, next_bbox, bbox_gt_list, hit_flags):
    """

    :param cur_bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param next_bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param bbox_gt_list: (list of tensor) [(4,)]
    :param hit_flags: (list)
    :return: reward (float) updated_hit_flags (list)
    """

    assert len(bbox_gt_list) > 0, "invalid reward bbox_gt_list input"

    # (2, 4)
    bbox_pred = torch.tensor([cur_bbox, next_bbox], dtype=bbox_gt_list[0].dtype).to(bbox_gt_list[0].device)

    # (M, 4)
    bbox_gt = torch.stack(bbox_gt_list, dim=0)

    bbox_iou = box_iou(bbox_pred, bbox_gt)  # (2, M)

    hit_flags = torch.tensor(hit_flags, dtype=torch.long)
    hit = (bbox_iou[1] > 0.5).cpu()
    if (hit.long() - hit_flags).max() > 0:
        # Stimulation (+5) given to those actions which cover any ground-truth objects
        # with an IoU greater than 0.5 for the first time.
        reward = 5.
        hit_flags[hit] = 1
    else:
        # if any ground-truth object bounding box has a higher IoU with the next window than the current one,
        # the reward of the action moving from the current window to the next one is +1, and −1 otherwise
        if ((bbox_iou[1] - bbox_iou[0]) > 0).any():
            reward = 1.
        else:
            reward = -1.

    return reward, hit_flags.tolist()
