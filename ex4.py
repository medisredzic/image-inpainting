import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch

def ex4(image_array: np.ndarray, offset: tuple, spacing: tuple):
    target = image_array.copy()
    known_array = np.zeros_like(image_array)
    known_array[offset[1]::spacing[1], offset[0]::spacing[0], :] = 1
    target_array = image_array[known_array == 0].copy
    image_array[known_array == 0] = 0

    # Target => Full Image, as passed to the function, without any editing.
    # Image_array => Image with applied offset and spacing to it(black grid)
    # Known_array => Full black image

    return target, image_array, known_array, target_array


if __name__ == '__main__':
    img = 'img.jpg'
    img = Image.open(img)
    img = np.asarray(img)

    t, ia, ka, k = ex4(img, (2, 6), (4, 8))

    #print(type(ia), type(ta))

    #print(ia.shape, ta.shape)
    #taa = np.transpose(ka, (1, 2, 0))
    taa = Image.fromarray(ia)
    taa.save('t.jpg')

