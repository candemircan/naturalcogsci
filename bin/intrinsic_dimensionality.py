from os.path import join
import os
import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from skdim.id import TwoNN

from naturalcogsci.helpers import get_project_root

def id_calculator(data):
    data = np.unique(data, axis=0)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    nn_method = TwoNN()
    return nn_method.fit_transform(data)


def main():
    project_root = get_project_root()

    feature_files = glob.glob(join(project_root, "data", "features", "*.npy"))

    ed_scores = []

    for feature_file in tqdm(feature_files):
        features = np.load(feature_file)
        feature_name = os.path.basename(feature_file).split(".npy")[0]

        file_name = join(project_root, "data", "ID", f"{feature_name}.csv")



        # Check if the file already exists
        if not os.path.exists(file_name):
            local_id = id_calculator(features)
            ed_scores ={
                    "Feature": feature_name,
                    "local ID": [local_id],
            }
            df = pd.DataFrame(ed_scores)
            df.to_csv(file_name, index=False)


        else:
            print(f"File {file_name} already exists, skipping.")


if __name__ == "__main__":
    main()
