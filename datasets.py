import os
import glob
import torch

from ex4 import ex4
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

IMG_SHAPE = 100
MIN_OFFSET = 0
MAX_OFFSET = 8
MIN_SPACING = 2
MAX_SPACING = 6
MIN_KNOWN_PIXELS = 144

SPACING = [2, 3, 4, 5, 6]
OFFSET = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class LoadImages(Dataset):
    def __init__(self, folder):
        self.input_abs = os.path.abspath(folder)
        self.files = []

        self.files = (os.path.join(f) for f in glob.glob(os.path.join(self.input_abs, '**', '*.jpg'), recursive=True))

        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        resize_transforms = transforms.Compose([
            transforms.Resize(size=IMG_SHAPE),
            transforms.CenterCrop(size=(IMG_SHAPE, IMG_SHAPE)),
        ])

        image = Image.open(self.files[idx]).convert('RGB')
        image = resize_transforms(image)

        image = np.array(image, dtype=np.float32)
        target, image_arr, known_arr, target_arr = ex4(image,
                                                     (random.choice(OFFSET), random.choice(OFFSET)),
                                                     (random.choice(SPACING), random.choice(SPACING)))

        image_arr = np.transpose(image_arr, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))

        return target, image_arr


def stack_with_padding(batch_as_list: list):
    n_samples = len(batch_as_list)
    
    image_shapes = np.stack([np.array(sample[0].shape) for sample in batch_as_list], axis=-1)
    max_image_shape = image_shapes.max(axis=-1)
    
    stacked_images = torch.full(size=(n_samples, *max_image_shape), dtype=torch.float32, fill_value=0.)
    for i in range(n_samples):
        stacked_images[i, :, :batch_as_list[i][0].shape[-2], :batch_as_list[i][0].shape[-1]] \
            = torch.from_numpy(batch_as_list[i][0])

    targets_list = [torch.from_numpy(sample[1]) for sample in batch_as_list]
    ids_list = [sample[2] for sample in batch_as_list]
    return stacked_images, targets_list, ids_list


if __name__ == '__main__':
    img_data = LoadImages('training/000')
    img_loader = DataLoader(img_data, shuffle=True, batch_size=5, collate_fn=stack_with_padding)

    for i, (inp, trg, ids) in enumerate(img_loader):
        print(f"Batch {i}:")
        print(f"image ids: {ids}")
        print(f"batch shape: {inp.shape}")