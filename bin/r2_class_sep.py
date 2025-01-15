from os.path import join, exists

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist, pdist


from naturalcogsci.helpers import get_project_root


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



def main(args):
    project_root = get_project_root()

    file_name = join(
        project_root, "data", "r2", f"{args.features.replace('/','_')}.csv"
    )

    print(file_name)

    if exists(file_name):
        print(f"{args.features} already extracted")
        return
    

    features = np.load(
        join(project_root, "data", "features", f"{args.features.replace('/','_')}.npy")
    )

    task_features = np.load(join(project_root, "data", "features", "task.npy"))
    unique_features, indices = np.unique(task_features, axis=0, return_inverse=True)
    class_labels = np.arange(len(unique_features))
    class_labels = class_labels[indices]

    r2 = class_separation(features, class_labels)
    df = pd.DataFrame({"r2": [r2]})
    df.to_csv(file_name, index=False)

    print(f"{args.features} class-separation done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", "-f")

    args = parser.parse_args()

    main(args)
