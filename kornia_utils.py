import kornia
import numpy as np
import matplotlib.pyplot as plt

def k_to_img_np(img_t):
    return kornia.tensor_to_image(img_t * 255.0).astype(dtype=np.uint8)


def show_torch_img(img_t, title, figsize=(6, 8), show_really=True):
    if show_really:
        img_np = k_to_img_np(img_t)
        plt.figure(figsize=figsize)
        plt.axis('equal')
        plt.title(title)
        plt.imshow(img_np)
        plt.show(block=False)
