"""
Plot picture and bounding box based on results(shuffle=False)
"""
import argparse, json, math, os

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataloader import VOCLocalization


def visualize_tree(args):

    print('[INFO]: visualizing {} images of {} on {} set'.format(args['num_images'] if args['num_images'] > 0 else 'all',
                                                                  args['json_path'], args['dataset']))

    # dataset
    voc_loader = VOCLocalization(args['voc_path'], year='2007', image_set=args['dataset'], download=False,
                                 transform=VOCLocalization.transform_for_img(args['max_size']))

    # result json
    with open(args['json_path'], 'r') as f:
        results = json.load(f)
    all_bbox_pred = results['bbox']
    vis_path = os.path.join(os.path.split(args['json_path'])[0], args['vis_dir'])
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    print('[INFO]: visual tree jpg save to', vis_path)

    assert len(voc_loader) == len(all_bbox_pred), "Dataset and Result json mismatch"

    bbox_num = len(all_bbox_pred[0])  # for example: 15
    n_cols = (bbox_num + 1) // 2  # 8
    n_rows = int(math.log(n_cols, 2)) + 1  # 4

    for (img, _, bboxes_gt, idx), bboxes_pred in tqdm(zip(voc_loader, all_bbox_pred),
                                                      total=min(args['num_images'], len(voc_loader))
                                                      if args['num_images'] > 0 else len(voc_loader)):

        if idx >= args['num_images'] > 0:
            break

        fig, axes = plt.subplots(n_rows, n_cols)
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r][c]
                ax.set_axis_off()
                if c < 2 ** r:
                    bbox_img = img.copy()
                    draw = ImageDraw.Draw(bbox_img)
                    draw.rectangle(bboxes_pred[2 ** r + c - 1], outline='red', width=5)
                    if 2 ** r + c - 1 == 0: # add ground truth to root
                        for bbox_gt in bboxes_gt:
                            draw.rectangle(bbox_gt.tolist(), outline='blue', width=5)

                    ax.imshow(bbox_img)

        fig.suptitle('bbox Tree')
        fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0)
        fig.savefig(os.path.join(vis_path, 'idx_{}.jpg'.format(idx)), dpi=1000)
        plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_size", type=int,
                        help="longest edge of the image")
    parser.add_argument("--json_path", type=str, help="the path of result json")
    parser.add_argument("--voc_path", type=str, default="./data/voc2007",
                        help="the root directory of data")
    parser.add_argument("--vis_dir", type=str, default="vis",
                        help="the sub directory name to save images")
    parser.add_argument("--num_images", type=int, default=-1,
                        help="num images to visual -1 for all")
    parser.add_argument("--dataset", type=str, default="test",
                        help="which dataset to evaluate")
    visualize_args = vars(parser.parse_args())

    visualize_tree(visualize_args)
