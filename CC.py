import os
import numpy as np
import cv2

# Define folders
ground_truth_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMapsOutput"

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

# Get all ground truth filenames
ground_truth_files = sorted(os.listdir(ground_truth_folder))

# Compute CC for all images
cc_scores = []
total_images = len(ground_truth_files)

for idx, filename in enumerate(ground_truth_files):
    ground_truth_path = os.path.join(ground_truth_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        ground_truth_map = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        # Compute CC score
        score = cc(ground_truth_map, saliency_map)

        if not np.isnan(score):
            cc_scores.append(score)

    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Compute mean CC score
mean_cc = np.mean(cc_scores) if cc_scores else np.nan
print(f"Mean CC Score: {mean_cc:.4f}")
