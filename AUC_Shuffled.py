import numpy as np
import cv2
import matplotlib.pyplot as plt


def auc_shuffled(saliency_map, fixation_map, other_map, n_splits=100, step_size=0.1, to_plot=False):
    """
    Computes the shuffled AUC score for saliency prediction.

    :param saliency_map: The saliency map (2D NumPy array)
    :param fixation_map: The human fixation map (binary 2D NumPy array)
    :param other_map: Binary fixation map from other images
    :param n_splits: Number of random splits (default: 100)
    :param step_size: Step size for threshold sweeping (default: 0.1)
    :param to_plot: Whether to plot the ROC curve (default: False)
    :return: AUC score
    """
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

    Sth = S[F > 0]  # Saliency values at fixation locations
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

    score = np.mean(auc_scores)

    if to_plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(saliency_map, cmap='gray')
        plt.scatter(*np.where(fixation_map > 0)[::-1], c='r', s=5)
        plt.title("Saliency Map with Fixations")

        plt.subplot(1, 2, 2)
        plt.plot(fp, tp, 'b.-')
        plt.title(f"ROC Curve (AUC: {score:.4f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    return score
