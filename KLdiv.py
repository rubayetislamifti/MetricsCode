import numpy as np
from skimage.transform import resize

def kl_div(saliency_map, fixation_map):
    """
    Computes the KL-divergence between two saliency maps when viewed as probability distributions.

    Parameters:
    - saliency_map: The saliency map (grayscale image).
    - fixation_map: The human fixation map.

    Returns:
    - score: KL-divergence value.
    """
    # Resize saliency map to match fixation map size
    map1 = resize(saliency_map, fixation_map.shape, mode='reflect', anti_aliasing=True)
    map2 = fixation_map.astype(np.float64)  # Ensure it's a float array

    # Normalize to make sure maps sum to 1 (convert to probability distributions)
    if np.any(map1):
        map1 /= np.sum(map1)
    if np.any(map2):
        map2 /= np.sum(map2)

    # Compute KL-divergence
    score = np.sum(map2 * np.log(np.finfo(float).eps + map2 / (map1 + np.finfo(float).eps)))

    return score
