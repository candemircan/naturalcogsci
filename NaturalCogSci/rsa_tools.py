from __future__ import annotations


__all__ = ["cka", "class_separation"]

import numpy as np
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm


def cka(
    X: np.ndarray,  # Representations of the first set of samples.
    Y: np.ndarray,  # Representations of the second set of samples.
) -> float:  # The linear CKA between X and Y.
    """
    Compute the linear CKA between two matrices X and Y.

    [link to the paper](https://arxiv.org/abs/1905.00414)

    taken from Patrick Mineault's implementation of CKA as is.

    [link to original implementation](https://goodresearch.dev/cka.html)

    Matrices should be observations by features.

    """
    # Implements linear CKA as in Kornblith et al. (2019)
    X = X.copy()
    Y = Y.copy()

    # Center X and Y
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Calculate CKA
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)

    return (YTX**2).sum() / np.sqrt((XTX**2).sum() * (YTY**2).sum())


def class_separation(
    X: np.ndarray,  # Feature by observation representation matrix.
    classes: np.ndarray,  # Class labels for each observation.
) -> float:  # The class separation of X
    """
    Compute the class separation $R^2$ as defined in [this paper](https://arxiv.org/abs/2010.16402)
    """
    unique_classes = np.unique(classes)
    total_classes = len(unique_classes)
    d_within = 0
    d_total = 0


    # Compute the within class distance
    for cls_ in tqdm(unique_classes):
        class_examples = X[classes == cls_]
        total_examples = len(class_examples)
        # Compute pairwise cosine distances for examples in the same class
        pairwise_distances = pdist(class_examples, metric="cosine")
        # Sum up the distances and normalize
        d_within += pairwise_distances.sum() / (total_classes * total_examples**2)

    # Compute the total distance
    for cls_1 in tqdm(unique_classes):
        cls_1_examples = X[classes == cls_1]
        for cls_2 in unique_classes:
            cls_2_examples = X[classes == cls_2]
            # Compute pairwise cosine distances for examples from different classes
            pairwise_distances = cdist(cls_1_examples, cls_2_examples, metric="cosine")
            # Sum up the distances and normalize
            d_total += pairwise_distances.sum() / (
                total_classes**2 * len(cls_1_examples) * len(cls_2_examples)
            )

    return 1 - d_within / d_total
