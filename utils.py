#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import torchvision.transforms.functional as transforms
import numpy as np


def load_img(path):
    img = Image.open(path).convert('RGB')

    return transforms.to_tensor(img)
