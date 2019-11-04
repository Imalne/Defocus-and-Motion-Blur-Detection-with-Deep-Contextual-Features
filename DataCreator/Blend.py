# coding: utf-8
from DataCreator.DataGenerator import *
import random

def motion_blur(image,blur_kernel):
    image = np.array(image)
    blurred = cv2.filter2D(image, -1, blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def motion_blur_kernel(degree, angle):
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    return kernel


def normalvariate_random_int(mean, variance, dmin, dmax):
    r = dmax + 1
    while r < dmin or r > dmax:
        r = int(random.normalvariate(mean, variance))
    return r


def uniform_random_int(dmin, dmax):
    r = random.randint(dmin,dmax)
    return r


def random_blur_kernel(mean=50, variance=15, dmin=10, dmax=100):
    random_degree = normalvariate_random_int(mean, variance, dmin, dmax)
    random_angle = uniform_random_int(-180, 180)
    return motion_blur_kernel(random_degree,random_angle)


def alpha_blending(defocus,motion,alpha):
    f_defocus = defocus.astype("float32")
    f_motion = motion.astype("float32")
    f_blended = f_defocus*(1-alpha) + f_motion * alpha
    return f_blended.astype("uint8")


def blended_mask(defocus_mask,motion_blur_mask,shape):
    defocus_mask = cv2.resize(defocus_mask,shape)
    motion_blur_mask = cv2.resize(motion_blur_mask,shape)
    mask = np.zeros(defocus_mask.shape)[:, :, 0]
    defocus_mask[motion_blur_mask > 0] = 128
    mask[defocus_mask[:, :, 0] == 0] = 2
    mask[defocus_mask[:, :, 1] == 128] = 1
    mask[defocus_mask[:, :, 2] == 255] = 0
    return mask


def both_blur_image_creator(defocus, extract_object, defocus_mask, object_mask, shape, motion_blur_threshold=0):
    kernel = random_blur_kernel(15, 2)
    defocus = cv2.resize(defocus,shape)
    extract_object = cv2.resize(extract_object, shape)
    defocus_mask = cv2.resize(defocus_mask,shape)
    object_mask = cv2.resize(object_mask,shape)

    motion_blur_img = motion_blur(extract_object, kernel)
    blur_mask = motion_blur(object_mask,kernel)
    motion_blur_mask = np.copy(blur_mask)
    motion_blur_mask[blur_mask > motion_blur_threshold] = 255

    alpha = blur_mask / 255
    blended = alpha_blending(defocus, motion_blur_img, alpha)
    return blended, motion_blur_mask, defocus_mask


if __name__ == '__main__':
    img = cv2.imread('./0004_img.png')
    object = cv2.imread('./0004_object.png')
    mask = cv2.imread('./0004_mask.png')
    defocus = cv2.imread('./defocus.jpg')
    defocus_mask = cv2.imread('./defocus_mask.png')
    blended,motion_blur,motion_blur_mask,defocus_mask = both_blur_image_creator(defocus,object,defocus_mask,mask,(400,300))
    blended_mask = blended_mask(defocus_mask,motion_blur_mask,(400,300))
    cv2.imshow('mask', blended_mask)
    cv2.imshow('blended',blended)
    cv2.waitKey()
    exit(0)