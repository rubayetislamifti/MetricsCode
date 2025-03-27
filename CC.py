import numpy as np
import cv2


def cc(saliency_map1, saliency_map2):
    """
    Computes Pearson's correlation coefficient between two saliency maps.

    Parameters:
        saliency_map1 (numpy.ndarray): First saliency map.
        saliency_map2 (numpy.ndarray): Second saliency map.

    Returns:
        float: Correlation coefficient between the two maps.
    """
    # Resize saliency_map1 to match saliency_map2
    saliency_map1 = cv2.resize(saliency_map1, (saliency_map2.shape[1], saliency_map2.shape[0]))

    # Convert to double precision
    map1 = saliency_map1.astype(np.float64)
    map2 = saliency_map2.astype(np.float64)

    # Normalize the maps
    map1 = (map1 - np.mean(map1)) / np.std(map1)
    map2 = (map2 - np.mean(map2)) / np.std(map2)

    # Compute Pearson correlation coefficient
    score = np.corrcoef(map1.flatten(), map2.flatten())[0, 1]

    return score
