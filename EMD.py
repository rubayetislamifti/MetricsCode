import os
import cv2
import numpy as np
from pyemd import emd

# Define folders
ground_truth_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMapsOutput"

def compute_emd(saliency_map, fixation_map, downsize=32):
    """
    Compute the Earth Mover's Distance (EMD) between a saliency map and a fixation map.

    :param saliency_map: Grayscale image representing the saliency map.
    :param fixation_map: Grayscale image representing the fixation map.
    :param downsize: Factor to resize the images for efficiency.
    :return: emd_score
    """
    # Resize images for efficiency
    fixation_map_resized = cv2.resize(fixation_map, (fixation_map.shape[1] // downsize, fixation_map.shape[0] // downsize))
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

    return emd_score

# Get all ground truth filenames
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# Compute EMD for all images
emd_scores = []
total_images = len(ground_truth_files)

for idx, filename in enumerate(ground_truth_files):
    ground_truth_path = os.path.join(ground_truth_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        ground_truth_map = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        # Compute EMD score
        score = compute_emd(saliency_map, ground_truth_map)

        if not np.isnan(score):
            emd_scores.append(score)

    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Compute mean EMD score
mean_emd = np.mean(emd_scores) if emd_scores else np.nan
print(f"Mean EMD Score: {mean_emd:.4f}")
