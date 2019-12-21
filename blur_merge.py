import cv2
import argparse
from glob import glob
import os
import numpy as np

def merge_class(img, map):
    class_num = 0
    for i in map.keys():
        img[img == i] = map[i]
        if class_num < map[i]:
            class_num = map[i]
    return img, img*(255//(class_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_dir", '-p', type=str, required=True)
    parser.add_argument("--target_dir", '-t', type=str, required=True)
    parser.add_argument("--merge_map", '-m', type=str, required=True, nargs='+')
    parser.add_argument("--save_dir",'-s',type=str, default='./submit')
    args = parser.parse_args()
    map = {}
    for i in args.merge_map:
        a_cls,b_cls = int(str.split(i,'-')[0]), int(str.split(i,'-')[1])
        map[a_cls] = b_cls
    tar_paths = glob(os.path.join(args.target_dir, "*"))
    img_paths = glob(os.path.join(args.prediction_dir, "*"))
    save_gt_dir = os.path.join(args.save_dir,'merge_gt')
    save_pred_dir = os.path.join(args.save_dir,'merge_predict')
    save_view_dir = os.path.join(args.save_dir,'merge_view')
    os.makedirs(save_gt_dir, exist_ok=True)
    os.makedirs(save_pred_dir, exist_ok=True)
    os.makedirs(save_view_dir, exist_ok=True)

    view_preds = []
    view_tars = []

    for i in tar_paths:
        img = cv2.imread(i)
        image_name = os.path.basename(i)
        img,view = merge_class(img,map)
        img = img[:,:,0]
        view = view[:,:,0]
        view_tars.append(view)
        cv2.imwrite(os.path.join(save_gt_dir,image_name),img)

    for i in img_paths:
        img = cv2.imread(i)
        image_name = os.path.basename(i)
        img, view = merge_class(img, map)
        img = img[:, :, 0]
        view = view[:, :, 0]
        view_preds.append(view)
        cv2.imwrite(os.path.join(save_pred_dir,image_name),img)

    for i in zip(view_tars,view_preds,img_paths):
        view = np.vstack((i[0],i[1]))
        image_name = os.path.basename(i[2])
        cv2.imwrite(os.path.join(save_view_dir,image_name),view)