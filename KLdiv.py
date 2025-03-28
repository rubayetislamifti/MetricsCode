import os
import numpy as np
import cv2
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


# Define folders
ground_truth_folder = r"I:\Saliency4asd\Saliency4asd\ASD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\ASD_FixMapsOutput"

# Get all ground truth filenames
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# Compute KL-divergence for all images
kl_scores = []
total_images = len(ground_truth_files)

for idx, filename in enumerate(ground_truth_files):
    ground_truth_path = os.path.join(ground_truth_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        fixation_map = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        # Compute KL-divergence score
        score = kl_div(saliency_map, fixation_map)

        if not np.isnan(score):
            kl_scores.append(score)

    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Compute mean KL-divergence score
mean_kl = np.mean(kl_scores) if kl_scores else np.nan
print(f"Mean KL-Divergence Score: {mean_kl:.4f}")
