import numpy as np
import cv2
import torch


def affine_map(x, dom_min, dom_max, val_min, val_max):
    a = (val_max - val_min) / (dom_max - dom_min)
    b = val_min
    return a * (x - dom_min) + b


def lab_to_rgb(lab_images, value_range):
    lab_images = affine_map(torch.permute(lab_images, (0, 2, 3, 1)), value_range[0], value_range[1], 0., 255.)
    lab_images = lab_images.to(dtype=torch.uint8).cpu().detach().numpy()
    rgb_images = []
    for lab_image in lab_images:
        rgb_images.append(cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB))
    rgb_images = affine_map(
        torch.permute(torch.tensor(np.array(rgb_images)), (0, 3, 1, 2)), 0., 255, value_range[0], value_range[1]
    )
    return rgb_images