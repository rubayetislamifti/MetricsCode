import os
import numpy as np
import cv2
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


# Define folders
ground_truth_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMapsOutput"

# Get all ground truth filenames
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# Compute Information Gain for all images
ig_scores = []
total_images = len(ground_truth_files)

for idx, filename in enumerate(ground_truth_files):
    ground_truth_path = os.path.join(ground_truth_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        fixation_map = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        # Use a uniform baseline (e.g., mean saliency map)
        baseline_map = np.ones_like(fixation_map, dtype=np.float64)  # Uniform baseline

        # Compute Information Gain score
        score = info_gain(saliency_map, fixation_map, baseline_map)

        if not np.isnan(score):
            ig_scores.append(score)

    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Compute mean Information Gain score
mean_ig = np.mean(ig_scores) if ig_scores else np.nan
print(f"Mean Information Gain Score: {mean_ig:.4f}")
