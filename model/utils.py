import torch
import numpy as np
import cv2

def sub_mean(x):
    """
    Subtrai a média espacial de x em cada batch e canal,
    retornando também a própria média para colocar dnv mais tarde.
    x: [B, C, H, W]
    """
    mean = x.mean(dim=[2,3], keepdim=True)
    x_centered = x - mean
    return x_centered, mean

def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().clone()
    tensor = 0.5 * (tensor + 1.0)  # converte de [-1,1] para [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    arr = tensor.numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)

def resize_image_max_keep_ratio(image, max_w=1280, max_h=720):
    h, w = image.shape[:2]
    if (w > max_w) or (h > max_h):
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image
