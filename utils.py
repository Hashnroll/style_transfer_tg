#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import torchvision.transforms.functional as transforms
import numpy as np


IMG_SIZE = 1024
def load_img(path, new_size=IMG_SIZE):
    img = Image.open(path).convert('RGB')
    if new_size:
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(new_size * (height / width)), new_size)
        else:
            new_shape = (new_size, int(new_size * (width / height)))
        img = transforms.resize(img, new_shape, Image.BICUBIC)

    return transforms.to_tensor(img)