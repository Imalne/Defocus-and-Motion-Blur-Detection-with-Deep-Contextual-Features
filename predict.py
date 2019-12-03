from Net import Net,Net_Bn
import torch
import numpy as np
import cv2
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
import os
import tqdm
import argparse


class Predictor:
    def __init__(self, weight_path, has_BN=False):
        assert weight_path != ""
        if not has_BN:
            net = Net()
        else:
            net = Net_Bn()
        params = torch.load(weight_path)
        state_dict = params['model_state_dict']
        net.load_state_dict(state_dict)
        net.eval()
        self.model = net.cuda()
        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def predict(self, inp: str, target: str = None, merge_img=False):
        assert os.path.exists(inp)

        img = Image.open(inp, 'r')
        img_resize = cv2.resize(np.array(img), (224, 224))
        output, output_view = self.predict_(img)
        if target:
            tar = Image.open(target, 'r')
            resize = transforms.Resize((224,224))
            tar_resize = np.array(resize(tar))[:, :, np.newaxis]
            tar_resize_view = tar_resize.repeat(3, 2) * 127
            return np.hstack((img_resize, tar_resize_view, output_view)) if merge_img else output_view, output, tar_resize
        else:
            return np.hstack(img_resize, output_view) if merge_img else output_view, output, None

    def predict_(self, inp: np.array):
        org = np.array(inp)
        inp = self.trans(inp)
        inp = torch.unsqueeze(inp, 0).cuda()
        output = self.model(inp)[0].cpu().detach().numpy()[0]
        output = np.argmax(output, axis=0)[:, :, np.newaxis].repeat(3, 2)
        return output.astype(np.uint8)[:, :, 0], output.astype(np.uint8)*127

    def predict_dir(self, img_path, tar_path=None, out_dir="./submit2", merge_img=False):
        img_paths = glob(os.path.join(img_path, "*"))
        if tar_path:
            tar_paths = glob(os.path.join(tar_path, "*"))
            assert len(img_paths) == len(tar_paths)

            img_paths.sort()
            tar_paths.sort()
            bar = tqdm.tqdm(zip(img_paths, tar_paths), total=len(img_paths))

            try:
                for img, tar in bar:
                    print(img,tar)
                    view, output, target = self.predict(img, tar, merge_img=merge_img)

                    cv2.imwrite(os.path.join(out_dir, 'view', os.path.basename(tar)), view)
                    cv2.imwrite(os.path.join(out_dir, 'predict', os.path.basename(tar)), output)
                    cv2.imwrite(os.path.join(out_dir, 'gt', os.path.basename(tar)), target)
            except KeyboardInterrupt:
                bar.close()
                raise
            bar.close()
        else:
            bar = tqdm.tqdm(img_paths, total=len(img_paths))
            try:
                for img in bar:
                    view, output, _ = self.predict(img, merge_img=merge_img)
                    cv2.imwrite(os.path.join(out_dir, 'view', os.path.basename(tar_path)), view)
                    cv2.imwrite(os.path.join(out_dir, 'predict', os.path.basename(tar_path)), output)
            except KeyboardInterrupt:
                bar.close()
                raise
            bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--w_path", type=str, required=True)
    parser.add_argument("-b", "--bn", type=bool, default=False)
    parser.add_argument("-i", "--img_dir", type=str, required=True)
    parser.add_argument("-o", "--out_dir", type=str, default="./submit")
    parser.add_argument("-t", "--tar_dir", type=str, required=False)
    parser.add_argument("-m", "--merge_img", type=bool, default=False)
    args = parser.parse_args()
    predictor = Predictor(args.w_path,args.bn)
    predictor.predict_dir(args.img_dir,
                          args.tar_dir,
                          out_dir=args.out_dir,
                          merge_img=args.merge_img)
    # predictor = Predictor("mnet.pth")
    # predictor.predict_dir(img_path='C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_image',
    #                       tar_path='C:\\Users\\Whale\\Documents\\DataSets\\CUHK\\test_gt',
    #                       merge_img=True)
