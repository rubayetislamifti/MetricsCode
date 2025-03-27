import os
import numpy as np
import cv2
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


# Define the directories
ground_truth_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMapsOutput"

# Get all ground truth filenames
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# List to store NSS scores for each image
nss_scores = []
total_images = len(ground_truth_files)

# Loop through all files and compute NSS score
for idx, filename in enumerate(ground_truth_files):
    ground_truth_path = os.path.join(ground_truth_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        fixation_map = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        # Compute NSS score
        score = nss(saliency_map, fixation_map)

        if not np.isnan(score):
            nss_scores.append(score)

    # Display remaining images to be processed
    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Calculate the mean NSS score
mean_nss = np.mean(nss_scores) if nss_scores else np.nan
print(f"Mean NSS Score: {mean_nss:.4f}")
