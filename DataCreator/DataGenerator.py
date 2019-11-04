from DataCreator.Blend import *
import numpy as np
import os
import cv2

class Sampler:
    def __init__(self, defocus_img_dir, defocus_mask_dir, filte=None):
        self.image_path_list = []
        self.mask_path_list = []
        for _, _, file_names in os.walk(defocus_img_dir):
            for file in file_names:
                if filte is not None and not filte(file):
                    continue
                self.image_path_list.append(os.path.join(defocus_img_dir,file))
        for _, _, file_names in os.walk(defocus_mask_dir):
            for file in file_names:
                if filte is not None and not filte(file):
                    continue
                self.mask_path_list.append(os.path.join(defocus_mask_dir, file))

    def sample(self, num):
        index = np.array(range(len(self.image_path_list)))
        samples_index = np.random.choice(index,num,replace=False).tolist()
        samples_path = []
        for i in samples_index:
            samples_path.append((self.image_path_list[i],self.mask_path_list[i]))
        return samples_path


def defocus_filte(file_name):
    return file_name.find('out_of_focus', 0, len(file_name)) >= 0


if __name__ == '__main__':
    defocus_sampler = Sampler("C:\\Users\\Whale\\Documents\\LAB\\DataSet\\CUHK\\image",
                              "C:\\Users\\Whale\\Documents\\LAB\\DataSet\\CUHK\\gt",
                              defocus_filte)
    motion_sampler = Sampler("C:\\Users\\Whale\\Documents\\LAB\\DataSet\\HKU-IS\\objects",
                             "C:\\Users\\Whale\\Documents\\LAB\\DataSet\\HKU-IS\\gt")

    out_put_image_dir = "C:\\Users\\Whale\\Documents\\LAB\\DataSet\\SyncBlur\\image"
    out_put_mask_dir = "C:\\Users\\Whale\\Documents\\LAB\\DataSet\\SyncBlur\\gt2"

    count = 0
    d_samples = defocus_sampler.sample(num=564)
    for d_sample in d_samples:
        d_img = cv2.imread(d_sample[0])
        d_mask = cv2.imread(d_sample[1])
        m_samples = motion_sampler.sample(num=15)
        for m_sample in m_samples:
            m_img = cv2.imread(m_sample[0])
            m_mask = cv2.imread(m_sample[1])
            blended, motion_blur_mask, _ = both_blur_image_creator(np.copy(d_img), np.copy(m_img), np.copy(d_mask),np.copy(m_mask), (400, 300))
            b_mask = blended_mask(np.copy(d_mask), np.copy(motion_blur_mask), (400, 300))

            count += 1
            image_name = "{:0>5d}".format(count) + '.png'
            mask_name = "{:0>5d}".format(count) + '.png'
            cv2.imwrite(os.path.join(out_put_image_dir, image_name), blended)
            cv2.imwrite(os.path.join(out_put_mask_dir, mask_name), b_mask)

            print(count)
            print(d_sample[0])
            print(m_sample[0])


    pass