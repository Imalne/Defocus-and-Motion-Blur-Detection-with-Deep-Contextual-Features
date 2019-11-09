from Net import Net
from cfg import Configs
from DataSetLoader.MDataSet import BlurDataSet
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch


if __name__ == '__main__':
    model = Net().cuda()
    model.load_model(Configs['vgg_19_pre_path'], Configs['model_save_path'])
    test_dataset = BlurDataSet(data_dir=Configs["test_image_dir"], target_dir=Configs["test_mask_dir"])
    loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for _, (inp, tar) in enumerate(loader):
        image = np.transpose(inp[0][0].numpy(), (1, 2, 0))
        target = (tar[0][0].numpy().astype(np.uint8)*127).squeeze(0)

        input_image = torch.cat(inp, 0).cuda()
        prediction = model(input_image)[0].cpu()[0]
        prediction = (torch.argmax(prediction, dim=0).squeeze(0)).numpy().astype(np.uint8)*127

        cv2.imshow("image", image)
        cv2.imshow("target", target)
        cv2.imshow("prediction", prediction)
        cv2.waitKey(0)
    exit(0)