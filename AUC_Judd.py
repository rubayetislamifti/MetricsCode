import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# Define the folders
fixation_map_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMaps"
saliency_map_folder = r"I:\Saliency4asd\Saliency4asd\TD_FixMapsOutput"

# Function to compute AUC Judd
def auc_judd(saliency_map, fixation_map, jitter=True):
    if np.sum(fixation_map) == 0:
        return np.nan

    # Resize saliency_map to match fixation_map if needed
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, anti_aliasing=True)

    # Add jitter to avoid ties
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) / 10000000

    # Normalize the saliency map
    min_val, max_val = np.min(saliency_map), np.max(saliency_map)
    if max_val > min_val:
        saliency_map = (saliency_map - min_val) / (max_val - min_val)
    else:
        return np.nan

    S = saliency_map.ravel()
    F = fixation_map.ravel()

    Sth = S[F > 0]  # Saliency values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    all_thresholds = np.sort(Sth)[::-1]  # Sort in descending order
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[-1] = 1
    fp[-1] = 1

    for i, thresh in enumerate(all_thresholds, start=1):
        above_thresh = np.sum(S >= thresh)
        tp[i] = i / Nfixations  # True positive rate
        fp[i] = (above_thresh - i) / (Npixels - Nfixations)  # False positive rate

    score = np.trapz(tp, fp)
    return score

# Get all fixation map filenames
fixation_files = sorted(os.listdir(fixation_map_folder))

# Compute AUC Judd for all images
auc_scores = []
total_images = len(fixation_files)

for idx, filename in enumerate(fixation_files):
    fixation_path = os.path.join(fixation_map_folder, filename)
    saliency_path = os.path.join(saliency_map_folder, filename)

    if os.path.exists(saliency_path):
        fixation_map = imread(fixation_path, as_gray=True)  # Read ground truth
        saliency_map = imread(saliency_path, as_gray=True)  # Read predicted map

        score = auc_judd(saliency_map, fixation_map)
        if not np.isnan(score):
            auc_scores.append(score)

    remaining_images = total_images - (idx + 1)
    print(f"Remaining images: {remaining_images}")

# Compute mean AUC Judd score
mean_auc = np.mean(auc_scores) if auc_scores else np.nan
print(f"Mean AUC Judd Score: {mean_auc:.4f}")
