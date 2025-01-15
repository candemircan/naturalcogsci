from glob import glob
import numpy as np
import json
import pickle
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from os.path import basename

from naturalcogsci.helpers import get_project_root

PROJECT_ROOT = get_project_root()

representations = glob(f"{PROJECT_ROOT}/data/peterson_features/*npy")
a_file = open(f"{PROJECT_ROOT}/data/peterson/datasets_peterson.pkl", "rb")
datasets = pickle.load(a_file)
a_file.close()


file_names = open(f"{PROJECT_ROOT}/data/peterson_features/file_names.txt", "r")


# reading the file
file_names = file_names.read().strip()

# replacing end of line('/n') with ' ' and
# splitting the text it further when '.' is seen.
file_names = file_names.split("\n")
file_names = [basename(x) for x in file_names]

json_dict = {}
for representation_name in tqdm(representations):
    representation = np.load(representation_name)
    corrs = []
    for category in ["fruits", "vegetables", "animals"]:
        indices = [file_names.index(img) for img in datasets[category]["fnames"]]
        sim_matrix = cosine_similarity(representation[indices])
        model_sim_vector = sim_matrix[np.tril_indices(sim_matrix.shape[0], -1)]

        human_sim_vector = datasets[category]["similarity"][np.tril_indices(datasets[category]["similarity"].shape[0], -1)]

        corrs.append(spearmanr(human_sim_vector, model_sim_vector)[0])

    json_dict[basename(representation_name).split(".npy")[0]] = np.mean(corrs)

with open(f"{PROJECT_ROOT}/data/peterson/peterson_correlations.json", "w") as outfile:
    json.dump(json_dict, outfile)
