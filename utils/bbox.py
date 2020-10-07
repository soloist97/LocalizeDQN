ACTION_FUNC_DICT = {
    0 : lambda bbox : bbox_scaling(bbox, 0),  # top left
    1 : lambda bbox : bbox_scaling(bbox, 1),  # top right
    2 : lambda bbox : bbox_scaling(bbox, 2),  # down left
    3 : lambda bbox : bbox_scaling(bbox, 3),  # down right
    4 : lambda bbox : bbox_scaling(bbox, 4),  # middle
    5 : lambda bbox : bbox_translating(bbox, 0),  # right
    6 : lambda bbox : bbox_translating(bbox, 1),  # left
    7 : lambda bbox : bbox_translating(bbox, 2),  # down
    8 : lambda bbox : bbox_translating(bbox, 3),  # up
    9 : lambda bbox: bbox_stretching(bbox, 0),  # up down inside
    10 : lambda bbox: bbox_stretching(bbox, 1),  # left right inside
    11 : lambda bbox: bbox_stretching(bbox, 2),  # up down outside
    12 : lambda bbox: bbox_stretching(bbox, 3),  # left right outside
}


def bbox_scaling(bbox, direction, factor=0.55):
    """

    :param bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param direction: (int) [0, 4]
    :param factor:  (float) (0, 1]
    :return: new bbox (xmin', ymin', xmax', ymax')
    """

    width = bbox[2] - bbox[0]
    depth = bbox[3] - bbox[1]

    if direction == 0:  # top left
        new_bbox = (bbox[0], bbox[1], bbox[0] + width * factor, bbox[1] + depth * factor)
    elif direction == 1:  # top right
        new_bbox = (bbox[2] - width * factor, bbox[1], bbox[2], bbox[1] + depth * factor)
    elif direction == 2:  # down left
        new_bbox = (bbox[0], bbox[3] - depth * factor, bbox[0] + width * factor, bbox[3])
    elif direction == 3:  # down right
        new_bbox = (bbox[2] - width * factor, bbox[3] - depth * factor, bbox[2], bbox[3])
    elif direction == 4:  # middle
        new_bbox = (bbox[0] + (1 - factor) * width / 2, bbox[1] + (1 - factor) * depth / 2,
                    bbox[2] - (1 - factor) * width / 2, bbox[3] - (1 - factor) * depth / 2)
    else:
        new_bbox = bbox.copy()

    return new_bbox


def bbox_translating(bbox, direction, factor=0.25):
    """

    :param bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param direction: (int) [0, 3]
    :param factor:  (float) (0, 1]
    :return: new bbox (xmin', ymin', xmax', ymax')
    """

    width = bbox[2] - bbox[0]
    depth = bbox[3] - bbox[1]

    if direction == 0:  # right
        new_bbox = (bbox[0] + factor * width, bbox[1], bbox[2] + factor * width, bbox[3])
    elif direction == 1:  # left
        new_bbox = (bbox[0] - factor * width, bbox[1], bbox[2] - factor * width, bbox[3])
    elif direction == 2:  # down
        new_bbox = (bbox[0], bbox[1] + depth * factor, bbox[2], bbox[3] + depth * factor)
    elif direction == 3:  # up
        new_bbox = (bbox[0], bbox[1] - depth * factor, bbox[2], bbox[3] - depth * factor)
    else:
        new_bbox = bbox.copy()

    return new_bbox


def bbox_stretching(bbox, direction, factor=0.25):
    """

    :param bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param direction: (int) [0, 3]
    :param factor:  (float) (0, 1]
    :return: new bbox (xmin', ymin', xmax', ymax')
    """

    width = bbox[2] - bbox[0]
    depth = bbox[3] - bbox[1]

    if direction == 0:  # up down inside
        new_bbox = (bbox[0], bbox[1] + depth * factor, bbox[2], bbox[3] - depth * factor)
    elif direction == 1:  # left right inside
        new_bbox = (bbox[0] + width * factor, bbox[1], bbox[2] - width * factor, bbox[3])
    elif direction == 2:  # up down outside
        new_bbox = (bbox[0], bbox[1] - depth * factor, bbox[2], bbox[3] + depth * factor)
    elif direction == 3:  # left right outside
        new_bbox = (bbox[0] - width * factor, bbox[1], bbox[2] + width * factor, bbox[3])
    else:
        new_bbox = bbox.copy()

    return new_bbox


def next_bbox_by_action(cur_bbox, action, original_img_shape=None):
    """

    :param cur_bbox: (tuple) (xmin, ymin, xmax, ymax)
    :param action: (int)
    :param original_img_shape: (None or tuple) (weight, height)
    :return: next bbox (xmin', ymin', xmax', ymax')
    """

    global ACTION_FUNC_DICT

    new_bbox = ACTION_FUNC_DICT[action](cur_bbox) if action in ACTION_FUNC_DICT.keys() else cur_bbox

    # clip
    if isinstance(original_img_shape, tuple):
        new_bbox = (max(0, new_bbox[0]),
                    max(0, new_bbox[1]),
                    min(original_img_shape[0], new_bbox[2]),
                    min(original_img_shape[1], new_bbox[3]))

    return new_bbox

