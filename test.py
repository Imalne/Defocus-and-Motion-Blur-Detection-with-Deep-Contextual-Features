from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from MTransform import SyncRandomCrop
from Net import Net
from cfg import Configs
from DataSetLoader.MDataSet import BlurDataSet
from torch.utils.data import DataLoader


if __name__ == '__main__':
    model = Net().cuda()
    model.load_model(Configs['vgg_19_pre_path'], Configs['model_save_path'])
    test_dataset = BlurDataSet(data_dir=Configs["test_image_dir"],target_dir=Configs["test_mask_dir"])
    for inp, tar in enumerate(test_dataset):
        pass
    exit(0)