import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
os.environ["OMP_NUM_THREADS"] = "1"


def dropout_white_bg(image, threshold=240):
    """
    Dropout the white background from an image. A pixel is considered as background
    if the value of all RGB channels are greater than the threshold.

    Args:
    - image (PIL Image or Torch Tensor): The input image.
    - threshold (int): Threshold value to consider a pixel as white background.

    Returns:
    - Torch Tensor: Image tensor with white background dropped out.
    """
    if not TF._is_pil_image(image) and not isinstance(image, torch.Tensor):
        raise TypeError('Image should be PIL Image or torch.Tensor. Got {}'.format(type(image)))

    if TF._is_pil_image(image):
        image = TF.to_tensor(image)  # Convert PIL Image to torch Tensor

    # Mask where all channels have value above the threshold
    mask = (image > threshold / 255.0).all(0)

    # Set the pixel value to 0 (black) where the mask is true (white background)
    image[:, mask] = 0
    return image


def get_embedding(model, patches):
    embeddings = list()

    model.eval()
    with torch.no_grad():
        for patch in patches:
            patch = dropout_white_bg(patch, threshold=230)
            patch = patch.unsqueeze(0).cuda()
            embedding = model(patch)

            embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings


def get_pseudo_label(model, patches, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    scaler = StandardScaler()

    embeddings = get_embedding(model, patches)
    embeddings = scaler.fit_transform(embeddings)

    kmeans.fit(embeddings)
    labels = kmeans.labels_

    return labels
