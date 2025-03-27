import cv2
import numpy as np
from pyemd import emd
import matplotlib.pyplot as plt


def compute_emd(saliency_map, fixation_map, to_plot=False, downsize=32):
    """
    Compute the Earth Mover's Distance (EMD) between a saliency map and a fixation map.

    :param saliency_map: Grayscale image representing the saliency map.
    :param fixation_map: Grayscale image representing the fixation map.
    :param to_plot: Boolean, if True, display the maps and their histograms.
    :param downsize: Factor to resize the images for efficiency.
    :return: emd_score, distance_matrix, flow_matrix
    """
    # Resize images for efficiency
    fixation_map_resized = cv2.resize(fixation_map,
                                      (fixation_map.shape[1] // downsize, fixation_map.shape[0] // downsize))
    saliency_map_resized = cv2.resize(saliency_map, (fixation_map_resized.shape[1], fixation_map_resized.shape[0]))

    R, C = fixation_map_resized.shape

    # Normalize mass so that sum equals 1
    fixation_map_resized = fixation_map_resized / np.sum(fixation_map_resized)
    saliency_map_resized = saliency_map_resized / np.sum(saliency_map_resized)

    # Compute distance matrix
    D = np.zeros((R * C, R * C), dtype=np.float64)

    indices = np.array([(r, c) for r in range(R) for c in range(C)])
    for i, (r1, c1) in enumerate(indices):
        for j, (r2, c2) in enumerate(indices):
            D[i, j] = np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

    # Convert images to 1D vectors
    P = fixation_map_resized.flatten().astype(np.float64)
    Q = saliency_map_resized.flatten().astype(np.float64)

    # Compute Earth Mover's Distance
    emd_score = emd(P, Q, D)

    if to_plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(fixation_map, cmap='gray')
        axs[0, 0].set_title('Fixation Map')
        axs[0, 1].imshow(saliency_map, cmap='gray')
        axs[0, 1].set_title('Saliency Map')
        axs[1, 0].imshow(fixation_map_resized, cmap='gray')
        axs[1, 0].set_title(f'EMD Score: {emd_score:.4f}')
        axs[1, 1].imshow(saliency_map_resized, cmap='gray')
        axs[1, 1].set_title('Resized Saliency Map')
        plt.show()

    return emd_score, D
