#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import torchvision.transforms.functional as transforms
import numpy as np


def load_img(path, img_size=None):
    img = Image.open(path).convert('RGB')
    if img_size is not None:
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(img_size * (height / width)), img_size)
        else:
            new_shape = (img_size, int(img_size * (width / height)))
        img = transforms.resize(img, new_shape, Image.LANCZOS)

    return transforms.to_tensor(img)
