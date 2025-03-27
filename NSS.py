import numpy as np
from skimage.transform import resize

def nss(saliency_map, fixation_map):
    """
    Computes the Normalized Scanpath Saliency (NSS) score.

    Parameters:
    - saliency_map: The saliency map (grayscale image).
    - fixation_map: The human fixation map (binary matrix).

    Returns:
    - score: NSS value.
    """
    # Resize saliency map to match fixation map size
    map_resized = resize(saliency_map, fixation_map.shape, mode='reflect', anti_aliasing=True)

    # Normalize the saliency map
    map_norm = (map_resized - np.mean(map_resized)) / np.std(map_resized)

    # Mean value at fixation locations
    score = np.mean(map_norm[fixation_map.astype(bool)])

    return score
