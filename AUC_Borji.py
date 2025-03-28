import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Define paths
ground_truth_dir = r"I:\Saliency4asd\Saliency4asd\ASD_FixMaps"
saliency_maps_dir = r"I:\Saliency4asd\Saliency4asd\ASD_FixMapsOutput"

# List all images in both directories
gt_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
saliency_files = sorted([f for f in os.listdir(saliency_maps_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

def load_image_as_gray(image_path):
    """ Load image and convert it to grayscale. """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    return img

def auc_borji(saliency_map, fixation_map, nsplits=100, step_size=0.1, to_plot=False):
    """
    Compute the AUC Borji score.
    """
    if np.sum(fixation_map) <= 1:
        print("No fixations in fixation_map")
        return np.nan

    # Resize saliency map to match fixation map size
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[0]))

    # Normalize saliency map
    min_val, max_val = np.min(saliency_map), np.max(saliency_map)
    if max_val > min_val:
        saliency_map = (saliency_map - min_val) / (max_val - min_val)
    else:
        print("NaN saliency_map")
        return np.nan

    # Flatten matrices
    S = saliency_map.flatten()
    F = fixation_map.flatten()

    Sth = S[F > 0]  # Saliency values at fixation points
    n_fixations = len(Sth)
    n_pixels = len(S)

    # Sample random locations
    randfix = S[np.random.randint(0, n_pixels, (n_fixations, nsplits))]

    # Compute AUC for each random split
    auc_values = []
    for s in range(nsplits):
        curfix = randfix[:, s]
        all_threshes = np.linspace(max(np.concatenate((Sth, curfix))), 0, int(1 / step_size) + 2)

        tp = [0] + [np.sum(Sth >= t) / n_fixations for t in all_threshes] + [1]
        fp = [0] + [np.sum(curfix >= t) / n_fixations for t in all_threshes] + [1]

        auc_values.append(np.trapz(tp, fp))

    score = np.mean(auc_values)

    return score


# Process each pair of images
results = {}
total_images = len(gt_files)
for idx, (gt_file, saliency_file) in enumerate(zip(gt_files, saliency_files)):
    gt_path = os.path.join(ground_truth_dir, gt_file)
    saliency_path = os.path.join(saliency_maps_dir, saliency_file)

    # Load images
    fixation_map = load_image_as_gray(gt_path)
    saliency_map = load_image_as_gray(saliency_path)

    # Compute AUC Borji
    auc_score = auc_borji(saliency_map, fixation_map)

    # Store result
    results[gt_file] = auc_score

    # Print the progress
    print(f"Remaining image {idx + 1} of {total_images}")

# Calculate the mean AUC Borji score
mean_auc_borji = np.nanmean(list(results.values()))

# Print the mean AUC Borji score
print(f"Mean AUC Borji score: {mean_auc_borji:.4f}")
