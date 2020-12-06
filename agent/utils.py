import pathlib

import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from glob import glob


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=torchvision.transforms.ToTensor(), min_size=20):
        self.dataset_path = pathlib.Path(dataset_path)

        self.transform = transform
        self.tokens = []

        for run_path in sorted(self.dataset_path.glob('*')):
            for image_path in sorted(self.dataset_path.glob(f'{run_path.name}/images/*.png')):
                # 0_1.png
                self.tokens.append(f'{run_path.name}:{image_path.name}')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Get data
        token = self.tokens[idx].split(':')
        run = token[0]
        token = token[1]

        img = Image.open(self.dataset_path / f"{run}/images/{token}")
        img = self.transform(img)

        mask = Image.open(self.dataset_path / f"{run}/masks/{token}")
        mask = self.transform(mask)
        mask = torch.clamp(torch.round(mask), 0, 1)

        # Calculate masks for width and height
        width = torch.max(torch.sum(img, 1))
        width_mask = mask.clone()
        width_mask[width_mask == 1] = width
        # height = torch.max(torch.sum(img, 0))
        # height_mask = mask.clone()
        # height_mask[width_mask == 1] = height

        # return img, mask.squeeze(0), np.concatenate([width_mask, height_mask], 0)
        return img, mask.squeeze(0), width_mask.squeeze(0)


def accuracy(pred, label):
    return (pred == label).float().mean().cpu().detach().numpy()

def load_detection_data(dataset_path, num_workers=0, batch_size=32, drop_last=True, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=drop_last)
