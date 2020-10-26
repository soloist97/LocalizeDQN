import argparse, math, os

import h5py, pickle, torch
import numpy as np
from tqdm import tqdm

from model.encoder import RESNET50Encoder
from dataloader import DataLoaderPFG, VOCLocalization


@torch.no_grad()
def preprocess(args, device):

    encoder = RESNET50Encoder().to(device)
    encoder.eval()

    voc_loader = DataLoaderPFG(
                    VOCLocalization(args['voc_path'], year=args['year'], image_set=args['image_set'],
                                    download=False, transform=VOCLocalization.transform_for_tensor(args['max_size'])),
                    batch_size=args['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
                    collate_fn=VOCLocalization.collate_fn
                 )

    print('[INFO]: start preprocessing {} {}...'.format(args['year'], args['image_set']))
    feature_map_path = os.path.join(args['path'], 'feature_map_{}_{}.h5'.format(args['year'], args['image_set']))
    img_info_path = os.path.join(args['path'], 'img_info_{}_{}.pkl'.format(args['year'], args['image_set']))
    fm_size = math.ceil(args['max_size'] * encoder.spatial_scale)

    img_info_dict = {'gt_bbox': [], 'shape': []}
    with h5py.File(feature_map_path, 'w') as h:
        print('[INFO]: saving feature map to', feature_map_path)

        h.attrs['max_size'] = args['max_size']
        fm = h.create_dataset('feat', (len(voc_loader.dataset), encoder.num_outputs, fm_size, fm_size), dtype=np.float32)
        print('[INFO]: shape of feature map', fm.shape)

        for img_tensor, img_shape, bbox_gt_list, image_idx in tqdm(voc_loader, total=len(voc_loader)):

            img_tensor = img_tensor.to(device)

            feature_map = encoder.encode_image(img_tensor).cpu().numpy()

            for i in range(len(image_idx)):
                fm[image_idx[i]] = feature_map[i]
                img_info_dict['gt_bbox'].append([bbox_gt.numpy().astype(np.float32) for bbox_gt in bbox_gt_list[i]])
                img_info_dict['shape'].append(img_shape[i])

    print('[INFO]: saving image info to', img_info_path)
    with open(img_info_path, 'wb') as f:
        pickle.dump(img_info_dict, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str,
                        help="2007 or 2012")
    parser.add_argument("--image_set", type=str,
                        help="train, val, trainval, test")
    parser.add_argument("--max_size", type=int,
                        help="longest edge of the image")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for encoder to extract")
    parser.add_argument("--path", type=str, default='./data',
                        help="contain subdirectory voc2007 or voc2012 and store created files")
    parser.add_argument("--cpu", action='store_true', help="whether to use cpu")
    args_pp = vars(parser.parse_args())

    args_pp['voc_path'] = os.path.join(args_pp['path'], 'voc' + args_pp['year'])

    device = torch.device('cuda' if torch.cuda.is_available() and not args_pp['cpu'] else 'cpu')

    preprocess(args_pp, device)
