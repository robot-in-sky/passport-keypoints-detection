# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by Depu Meng (mdp@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
        
# Xiaochu Style
# (R,G,B)
color1 = [(240,2,127)] * 7

link_pairs1 = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5]]

point_color1 = [(240,2,127)] * 6

xiaochu_style = ColorStyle(color1, link_pairs1, point_color1)


# Chunhua Style
# (R,G,B)
color2 = [(252,176,243)] * 7

link_pairs2 = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 5]]

point_color2 = [(252,176,243)] * 6

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    # general
    parser.add_argument('--image-path',
                        help='Path of COCO val images',
                        type=str,
                        default='data/passport/images/valid/'
                        )

    parser.add_argument('--gt-anno',
                        help='Path of COCO val annotation',
                        type=str,
                        default='data/passport/annotations/person_keypoints_valid.json'
                        )

    parser.add_argument('--save-path',
                        help="Path to save the visualizations",
                        type=str,
                        default='visualization/coco/')

    parser.add_argument('--prediction',
                        help="Prediction file to visualize",
                        type=str,
                        required=True)

    parser.add_argument('--style',
                        help="Style of the visualization: Chunhua style or Xiaochu style",
                        type=str,
                        default='chunhua')

    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict


def plot(data, gt_file, img_path, save_path, 
         link_pairs, ring_color, save=True):

    # Load GT and predictions
    coco = COCO(gt_file)
    coco_dt = coco.loadRes(data)
    coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    coco_eval._prepare()
    gts_ = coco_eval._gts
    dts_ = coco_eval._dts

    p = coco_eval.params
    p.imgIds = sorted(set(p.imgIds))
    if p.useCats:
        p.catIds = sorted(set(p.catIds))
    p.maxDets = sorted(p.maxDets)

    threshold = 0.3
    joint_thres = 0.2
    point_radius = 12
    line_width = 6

    for catId in (p.catIds if p.useCats else [-1]):
        for imgId in p.imgIds:
            gts = gts_.get((imgId, catId), [])
            dts = dts_.get((imgId, catId), [])

            if not gts or not dts:
                continue

            dts = sorted(dts, key=lambda d: -d['score'])[:p.maxDets[-1]]
            img_name = str(imgId).zfill(12)
            img_file = os.path.join(img_path, f"{img_name}.jpg")

            data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if data_numpy is None:
                print(f"[WARN] Failed to read image: {img_file}")
                continue

            h, w = data_numpy.shape[:2]
            fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
            ax.imshow(data_numpy[:, :, ::-1])
            ax.axis('off')

            for gt in gts:
                vg = np.array(gt['keypoints'])[2::3]

                for dt in dts:
                    iou = compute_iou(gt['bbox'], dt['bbox'])
                    score = dt['score']
                    if iou < 0.1 or score < threshold:
                        continue

                    dt_joints = np.array(dt['keypoints']).reshape(-1, 3)
                    joints_dict = map_joint_dict(dt_joints)

                    # draw links
                    for i1, i2, color in link_pairs:
                        if (
                            i1 in joints_dict and i2 in joints_dict and
                            dt_joints[i1, 2] >= joint_thres and
                            dt_joints[i2, 2] >= joint_thres and
                            vg[i1] > 0 and vg[i2] > 0
                        ):
                            line = mlines.Line2D(
                                [joints_dict[i1][0], joints_dict[i2][0]],
                                [joints_dict[i1][1], joints_dict[i2][1]],
                                lw=line_width,
                                color=color
                            )
                            ax.add_line(line)

                    # draw points
                    for k, (x, y, v) in enumerate(dt_joints):
                        if v < joint_thres or vg[k] == 0 or x > w or y > h:
                            continue
                        circle = mpatches.Circle(
                            (x, y),
                            radius=point_radius,
                            ec='black',
                            fc=ring_color[k] if k < len(ring_color) else 'red',
                            linewidth=1
                        )
                        ax.add_patch(circle)

            if save:
                score_int = int(score * 1000)
                fname = f"score_{score_int}_id_{imgId}_{img_name}"
                fig.savefig(os.path.join(save_path, f"{fname}.jpg"), bbox_inches='tight', pad_inches=0)
                # fig.savefig(os.path.join(save_path, f"{imgId}.pdf"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)


def compute_iou(boxA, boxB):
    ax0, ay0, aw, ah = boxA
    bx0, by0, bw, bh = boxB
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / (union_area + np.spacing(1))

if __name__ == '__main__':

    args = parse_args()
    if args.style == 'xiaochu':
        # Xiaochu Style
        colorstyle = xiaochu_style
    elif args.style == 'chunhua':
        # Chunhua Style
        colorstyle = chunhua_style
    else:
        raise Exception('Invalid color style')
    
    save_path = args.save_path
    img_path = args.image_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception:
            print('Fail to make {}'.format(save_path))

    
    with open(args.prediction) as f:
        data = json.load(f)
    gt_file = args.gt_anno
    plot(data, gt_file, img_path, save_path, colorstyle.link_pairs, colorstyle.ring_color, save=True)

