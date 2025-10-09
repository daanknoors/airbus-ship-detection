
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

def plot_random_image(dir_img, ax=None, random_state=None, verbose=True):
    """Plot a random image from a directory."""
    if random_state is not None:
        random.seed(random_state)

    img_name = random.choice(os.listdir(dir_img))
    img = Image.open(dir_img / img_name)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(img_name)
    if ax is None:
        plt.show()
    if verbose:
        print (f"Path: {dir_img / img_name}, Image name: {img_name}, size: {img.size}, mode: {img.mode}")
    return ax
