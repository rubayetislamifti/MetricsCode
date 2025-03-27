import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def auc_judd(saliency_map, fixation_map, jitter=True, to_plot=False):
    """
    Computes the AUC Judd score to measure how well the saliency map predicts human fixations.

    :param saliency_map: 2D numpy array, the predicted saliency map
    :param fixation_map: 2D binary numpy array, the ground truth fixation map
    :param jitter: Boolean, whether to add small noise to avoid ties
    :param to_plot: Boolean, whether to plot the ROC curve
    :return: score (AUC), tp (true positive rates), fp (false positive rates), all_thresholds (used thresholds)
    """
    if np.sum(fixation_map) == 0:
        print("No fixationMap")
        return np.nan, None, None, None

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
        print("NaN saliencyMap")
        return np.nan, None, None, None

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
    all_thresholds = np.concatenate(([1], all_thresholds, [0]))

    if to_plot:
        plt.subplot(1, 2, 1)
        plt.imshow(saliency_map, cmap='gray')
        plt.title("SaliencyMap with fixations")
        y, x = np.where(fixation_map)
        plt.scatter(x, y, color='red', s=5)

        plt.subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        plt.title(f"Area under ROC curve: {score:.4f}")
        plt.show()

    return score, tp, fp, all_thresholds
