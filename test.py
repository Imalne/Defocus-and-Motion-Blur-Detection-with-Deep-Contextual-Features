from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from MTransform import SyncRandomCrop

if __name__ == '__main__':
    # syncCrop = SyncRandomCrop((200, 300))
    # crop_1 = transforms.Compose([syncCrop,transforms.RandomRotation((90,90))])
    # image_1 = Image.open("C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_image\\motion0003.jpg", "r")
    # image_2 = Image.open("C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_image\\motion0003.jpg", "r")
    #
    # while True:
    #     cropped_1= crop_1(image_1)
    #     syncCrop.rand_fix()
    #     cropped_2 = crop_1(image_2)
    #     syncCrop.rand_active()
    #     plt.subplot(1,2,1)
    #     plt.imshow(cropped_1)
    #     plt.subplot(1,2,2)
    #     plt.imshow(cropped_2)
    #     plt.show()
    #     plt.waitforbuttonpress()
    exit(0)