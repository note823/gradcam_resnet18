from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import os
from glob import glob


class CustomDataset(Dataset):
    def __init__(self, cfg_data_root, transform=None, shuffle=False):
        super(CustomDataset, self).__init__()

        self.img_list = sorted(glob(os.path.join(cfg_data_root, '*.*')))
        self.len = len(self.img_list)
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        img = img.unsqueeze(0)

        return img

    def __len__(self):
        return self.len


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        # Before resizing, transforming to PIL
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
