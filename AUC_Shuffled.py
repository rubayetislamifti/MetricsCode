import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define folders
fixation_map_folder = r"I:\Saliency4asd\Saliency4asd\ASD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\ASD_FixMapsOutput"

def auc_shuffled(saliency_map, fixation_map, other_map, n_splits=100, step_size=0.1, to_plot=False):
    if np.sum(fixation_map) == 0:
        print("No fixationMap")
        return np.nan

    # Resize saliency map to match fixation map size
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]))

    # Normalize saliency map
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    if np.isnan(saliency_map).all():
        print("NaN saliencyMap")
        return np.nan

    S = saliency_map.flatten()
    F = fixation_map.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]
    n_fixations = len(Sth)

    # Get random fixation locations from other images
    other_fixation_indices = np.where(Oth > 0)[0]
    n_fixations_oth = min(n_fixations, len(other_fixation_indices))
    randfix = np.zeros((n_fixations_oth, n_splits))

    for i in range(n_splits):
        np.random.shuffle(other_fixation_indices)
        randfix[:, i] = S[other_fixation_indices[:n_fixations_oth]]

    # Compute AUC for each random split
    auc_scores = []
    for s in range(n_splits):
        curfix = randfix[:, s]
        all_threshes = np.arange(0, max(np.max(Sth), np.max(curfix)) + step_size, step_size)[::-1]

        tp = np.zeros(len(all_threshes) + 2)
        fp = np.zeros(len(all_threshes) + 2)
        tp[0], tp[-1] = 0, 1
        fp[0], fp[-1] = 0, 1

        for i, thresh in enumerate(all_threshes):
            tp[i + 1] = np.sum(Sth >= thresh) / n_fixations
            fp[i + 1] = np.sum(curfix >= thresh) / n_fixations_oth

        auc_scores.append(np.trapz(tp, fp))

    return np.mean(auc_scores)

# Get all fixation map filenames
fixation_files = sorted(os.listdir(fixation_map_folder))

# Compute AUC Shuffled for all images
auc_scores = []
total_images = len(fixation_files)

for idx, filename in enumerate(fixation_files):
    fixation_path = os.path.join(fixation_map_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        fixation_map = cv2.imread(fixation_path, cv2.IMREAD_GRAYSCALE)
        saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)

        # Using the same fixation map as the other_map for simplicity
        score = auc_shuffled(saliency_map, fixation_map, fixation_map)

        if not np.isnan(score):
            auc_scores.append(score)

    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Compute mean AUC Shuffled score
mean_auc_shuffled = np.mean(auc_scores) if auc_scores else np.nan
print(f"Mean AUC Shuffled Score: {mean_auc_shuffled:.4f}")
