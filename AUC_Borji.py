import numpy as np
import cv2
import matplotlib.pyplot as plt


def auc_borji(saliency_map, fixation_map, nsplits=100, step_size=0.1, to_plot=False):
    """
    Compute the AUC Borji score.

    Parameters:
        saliency_map (numpy.ndarray): The saliency map.
        fixation_map (numpy.ndarray): The human fixation map (binary matrix).
        nsplits (int): Number of random splits.
        step_size (float): Step size for sweeping through saliency map.
        to_plot (bool): If True, plots the ROC curve.

    Returns:
        float: AUC Borji score.
    """
    if np.sum(fixation_map) <= 1:
        print("No fixations in fixation_map")
        return np.nan

    # Resize saliency map to match fixation map size if necessary
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

    if to_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(saliency_map, cmap='gray')
        plt.scatter(*np.where(fixation_map > 0)[::-1], color='red', s=10)
        plt.title("Saliency Map with Fixations")

        plt.subplot(1, 2, 2)
        plt.plot(fp, tp, 'b-')
        plt.title(f"Area under ROC curve: {score:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    return score
