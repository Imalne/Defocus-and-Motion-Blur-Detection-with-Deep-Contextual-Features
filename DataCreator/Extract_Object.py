import numpy as np
import os
import cv2


def object_extract(mask,img):

    extracted = np.copy(img)
    con = (mask == 0)
    extracted[con] = 0
    # cv2.normalize(extracted, extracted, 0, 255, cv2.NORM_MINMAX)
    return extracted.astype(np.uint8)


def extract(image_dir, mask_dir, out_dir):
    for _,_,file_names in os.walk(image_dir):
        for fileName in file_names:
            img_path = os.path.join(image_dir, fileName)
            mask_path = os.path.join(mask_dir, fileName)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            out_path = os.path.join(out_dir, fileName)
            out = object_extract(mask,img)
            cv2.imwrite(out_path, out)
            print(fileName + ' extract complete')

if __name__ == '__main__':
    p1 = 'C:\\Users\\Whale\\Documents\\LAB\\DataSet\\HKU-IS\\imgs'
    p2 = 'C:\\Users\\Whale\\Documents\\LAB\\DataSet\\HKU-IS\\gt'
    p3 = 'C:\\Users\\Whale\\Documents\\LAB\\DataSet\\HKU-IS\\objects'
    extract(p1, p2, p3)