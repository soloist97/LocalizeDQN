import torch
from torchvision.ops.boxes import box_iou


class bboxPrecisionRecall(object):

    def __init__(self, IoU_thresholds=(0.5, 0.6, 0.7)):

        # True Positive (TP): A correct detection. Detection with IOU >= threshold
        # False Positive (FP): A wrong detection. Detection with IOU < threshold
        # False Negative (FN): A ground truth not detected

        self.results = {thr : {'TP': 0, 'FP': 0, 'FN': 0} for thr in IoU_thresholds}
        self.total = 0

    def update(self, bbox_gt, bbox_pred):
        """

        :param bbox_gt: (list[tensor]) N bbox with shape (4,)  dtype=torch.float (xmin, ymin, xmax, ymax) format
        :param bbox_pred: (list[tensor]) M bbox with shape (4, ) dtype=torch.float
        :return:
        """
        self.total += 1

        if len(bbox_gt) == 0:
            return
        elif len(bbox_pred) == 0:
            for thr in self.results.keys():
                self.results[thr]['FN'] += len(bbox_gt)
            return

        N = len(bbox_gt)
        M = len(bbox_pred)

        bbox_gt = torch.stack(bbox_gt, dim=0)  # (N, 4)
        bbox_pred = torch.stack(bbox_pred, dim=0)  # (M, 4)
        bbox_iou = box_iou(bbox_gt, bbox_pred)  # (N, M)

        iou_idx_pair = [(i, j) for i in range(N) for j in range(M)]
        iou_sorted_idx = bbox_iou.view(-1).argsort(descending=True).tolist()

        gt_match_idx = {thr: list() for thr in self.results.keys()}
        pred_match_idx = {thr: list() for thr in self.results.keys()}

        for idx in iou_sorted_idx:

            gidx, pidx = iou_idx_pair[idx]

            for thr in self.results.keys():
                if(bbox_iou[gidx][pidx] >= thr and gidx not in gt_match_idx[thr] and pidx not in pred_match_idx[thr]):
                    gt_match_idx[thr].append(gidx)
                    pred_match_idx[thr].append(pidx)

        for thr in self.results.keys():
            self.results[thr]['TP'] += len(gt_match_idx[thr])
            self.results[thr]['FP'] += M - len(pred_match_idx[thr])
            self.results[thr]['FN'] += N - len(gt_match_idx[thr])

    def clear(self):

        for thr in self.results.keys():
            self.results[thr] = {'TP': 0, 'FP': 0, 'FN': 0}
        self.total = 0

    def evaluate(self):

        pr_result = {thr: {'P': 0., 'R': 0.} for thr in self.results.keys()}
        if self.total == 0:
            return pr_result
        else:
            for thr in self.results.keys():

                # Precision = TP / (TP + FP)
                try:
                    pr_result[thr]['P'] = self.results[thr]['TP'] / (self.results[thr]['TP'] + self.results[thr]['FP'])
                except ZeroDivisionError:
                    pr_result[thr]['P'] = 0.

                # Recall = TP / (TP + FN)
                try:
                    pr_result[thr]['R'] = self.results[thr]['TP'] / (self.results[thr]['TP'] + self.results[thr]['FN'])
                except ZeroDivisionError:
                    pr_result[thr]['R'] = 0.

            return pr_result
