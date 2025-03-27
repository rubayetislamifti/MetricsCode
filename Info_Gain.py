import numpy as np
from skimage.transform import resize


def info_gain(saliency_map, fixation_map, baseline_map):
    """
    Computes the information gain of a saliency map over a baseline map.

    Parameters:
    - saliency_map: The saliency map (grayscale image).
    - fixation_map: The human fixation map (binary matrix).
    - baseline_map: Another saliency map (e.g., all fixations from other images).

    Returns:
    - score: Information gain value.
    """
    # Resize maps to match the fixation map size
    map1 = resize(saliency_map, fixation_map.shape, mode='reflect', anti_aliasing=True)
    mapb = resize(baseline_map, fixation_map.shape, mode='reflect', anti_aliasing=True)

    # Normalize and vectorize saliency maps
    map1 = (map1 - np.min(map1)) / (np.max(map1) - np.min(map1) + 1e-10)
    mapb = (mapb - np.min(mapb)) / (np.max(mapb) - np.min(mapb) + 1e-10)

    # Convert to probability distributions
    map1 /= np.sum(map1)
    mapb /= np.sum(mapb)

    # Get locations of fixations
    locs = fixation_map.astype(bool)

    # Compute information gain
    score = np.mean(np.log2(map1[locs] + np.finfo(float).eps) - np.log2(mapb[locs] + np.finfo(float).eps))

    return score
