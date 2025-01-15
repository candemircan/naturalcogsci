import os
from os.path import join
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from pathlib import Path

from naturalcogsci.helpers import get_project_root


def load_file_names(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def get_similarities(embeddings, ref_idx, left_idx, right_idx):
    ref_embed = embeddings[ref_idx].reshape(1, -1)
    left_embed = embeddings[left_idx].reshape(1, -1)
    right_embed = embeddings[right_idx].reshape(1, -1)

    left_sim = cosine_similarity(ref_embed, left_embed)[0][0]
    right_sim = cosine_similarity(ref_embed, right_embed)[0][0]

    return left_sim, right_sim


def compare_preferences(left_sim, right_sim, human_choice):
    model_choice = "left" if left_sim > right_sim else "right"
    return model_choice == human_choice


def process_embedding(embedding_path, df, file_to_index):
    embeddings = np.load(embedding_path)
    agreements = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {Path(embedding_path).stem}"):
        ref_idx = file_to_index[row["ref_path"]]
        left_idx = file_to_index[row["left_path"]]
        right_idx = file_to_index[row["right_path"]]

        left_sim, right_sim = get_similarities(embeddings, ref_idx, left_idx, right_idx)

        human_choice = "left" if row["left_vote"] == 1 else "right"

        agreement = compare_preferences(left_sim, right_sim, human_choice)
        agreements.append(agreement)

    agreement_rate = sum(agreements) / len(agreements)
    return agreement_rate


def main():
    df = pd.read_csv(join(PROJECT_ROOT, "data", "nights", "data.csv"))
    df = df[(df.votes >= 6) & (~df.is_imagenet) & (df.split == "test")].reset_index(drop=True)

    with open(f"{PROJECT_ROOT}/data/nights_features/file_names.txt", "r") as f:
        file_names = [line.strip().split("/lustre/groups/hcai/workspace/can.demircan/things_nights/dataset/nights/")[-1] for line in f]
    
    file_to_index = {name: i for i, name in enumerate(file_names)}


    # Create a dictionary mapping file names to their index in the embeddings array
    file_to_index = {name: i for i, name in enumerate(file_names)}

    results = {}
    features_folder = join(PROJECT_ROOT, "data", "nights_features")
    for filename in tqdm(os.listdir(features_folder), desc="Processing models"):
        if filename.endswith(".npy"):
            embedding_path = os.path.join(features_folder, filename)
            key = os.path.splitext(filename)[0]
            agreement_rate = process_embedding(embedding_path, df, file_to_index)
            results[key] = agreement_rate
            tqdm.write(f"{key}: {agreement_rate:.3f}")

    with open(f"{PROJECT_ROOT}/data/nights/nights.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    PROJECT_ROOT = get_project_root()
    main()
