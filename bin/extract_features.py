import argparse
import json
from os.path import join

import torch

from naturalcogsci.feature_extractors import extract_features
from naturalcogsci.helpers import str2bool, get_project_root

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--featurename", "-f", nargs="+")
    parser.add_argument("--cached", "-c", type=str2bool)

    args = parser.parse_args()

    if args.featurename == "all":
        project_root = get_project_root()
        with open(join(project_root, "data", "model_configs.json")) as f:
            features = json.load(f).keys()

    else:
        features = args.featurename

    for feature in features:
        print(f"Extracting features for {feature}", flush=True)
        torch.cuda.empty_cache()
        extract_features(feature, args.cached)
