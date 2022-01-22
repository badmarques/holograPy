import numpy as np
import torch
import os.path

from holograpy import settings
from holograpy.utils.logger import set_logger

from torch.utils.data import Dataset
from glob import glob
from os.path import join
from imageio import imread


def get_image_url(folder_path, fileformat = 'png'):
    files = glob( join(folder_path, "**/*." + fileformat), recursive=True)
    return files

class ClassificationDataset(Dataset):

    def __init__(self, split):

        self.logger = set_logger("ClassificationDataset")
        path = settings.DATASET_PATH[split]

        self.img_urls  = get_image_url(path, 'png')

    def __len__(self):
        dataset_size = len(self.img_urls)
        self.logger.debug(f"ClassificationDataset size is: {dataset_size}")
        return dataset_size

    def __getitem__(self, i):
        image = torch.Tensor(imread(self.img_urls[i]).astype(np.float32))/ 255.0
        image = image.expand((1, 224, 224))
        label = 0

        return (image, label)