from __future__ import annotations

__all__ = [
    "get_project_root",
    "prepare_training",
    "str2bool",
    "id_generator",
    "parse_reward_data",
    "parse_category_data",
    "filter_chance",
]

import json
import argparse
import string
import glob
import os
from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd

def get_project_root() -> str:  # project root
    """
    Return project root based on device.

    Reads the NATURALCOGSCI_ROOT environment variable

    Returns:
        str: project root
    """

    return os.getenv("NATURALCOGSCI_ROOT")


def prepare_training(task: str, features: str, cond_file: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares the observations and the target values to train models on,
    for the given condition file and the given task. The returned arrays
    have the shapes shown in the tables below.

    Args:
        task (str): 'reward_learning' or 'category_learning'
        features (str): which embedding to use. must match the saved .txt files
        cond_file (int): number of the condtion file to prepare arrays for

    Returns:
        Tuple[np.ndarray, np.ndarray]: X,y arrays
    """

    tasks = ["reward_learning", "category_learning"]
    assert task in tasks, f"{task} must be one of {tasks}"

    project_root = get_project_root()

    df = pd.read_csv(join(project_root, "data", "human_behavioural", task, "above_chance.csv"))
    df = df[df.cond_file == cond_file].reset_index(drop=True)

    with open(join(project_root, "data", "features", "file_names.txt"), "r") as f:
        file_names = f.read()

    file_names = file_names.split("\n")[:-1]
    file_names = [file_name.split("naturalcogsci/")[1] for file_name in file_names]
    embedding = np.load(join(project_root, "data", "features", f"{features}.npy"))
    if task == "reward_learning":
        TRIALS = 60
        OPTIONS = 2

        left_stimuli = df.left_image.to_list()
        right_stimuli = df.right_image.to_list()

        left_stimuli = [
            file_names.index(left_stimulus) for left_stimulus in left_stimuli
        ]
        right_stimuli = [
            file_names.index(right_stimulus) for right_stimulus in right_stimuli
        ]

        X = np.zeros((TRIALS, OPTIONS, embedding.shape[1]))
        X[:, 0, :] = embedding[left_stimuli, :]
        X[:, 1, :] = embedding[right_stimuli, :]

        y = np.zeros((TRIALS, OPTIONS))
        y[:, 0] = df.left_reward.to_list()
        y[:, 1] = df.right_reward.to_list()

    elif task == "category_learning":
        TRIALS = 120

        stimuli = df.image.to_list()[:TRIALS]

        stimuli = [file_names.index(stimulus) for stimulus in stimuli]

        X = np.zeros((TRIALS, embedding.shape[1]))
        X[:] = embedding[stimuli, :]
        y = df.true_category_binary.to_numpy()[:TRIALS]

    return X, y

def str2bool(v: str | bool) -> bool:
    """
    Used to parse boolean CLI arguments

    Args:
        v (str | bool): input from the command line

    Raises:
        argparse.ArgumentTypeError: When no boolean is passed

    Returns:
        bool: python bool
    """
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def id_generator(
    size: int = 10,  # length of ID
    chars: str = string.ascii_uppercase + string.digits,  # characters to use
) -> str:  # unique ID
    """
    Used to generate IDs for participants. This is to hide their Prolific IDs.
    """
    return "".join(np.random.choice(list(chars), size=size))


def parse_reward_data() -> None:
    """
    Read and parse json data and condition files of reward learning into pandas dataframes.

    Make 2 dataframes:

    1. all_df: includes all participants
    2. keep_df: includes only those with above 50% accuracy
    """

    project_root = get_project_root()
    BASE_PAY = 2.0
    TRIAL_NO = 60

    beh_files = glob.glob(
        join(project_root, "experiments", "reward_learning", "data", "*json")
    )
    df_list = []

    for beh_file_path in beh_files:
        with open(beh_file_path) as f:
            beh_file = json.load(f)

        beh_file_no = os.path.splitext(os.path.basename(beh_file_path))[0]

        with open(
            join(
                project_root,
                "experiments",
                "reward_learning",
                "condition_files",
                f"{beh_file_no}.json",
            )
        ) as f:
            cond_file = json.load(f)

        par_df = pd.DataFrame(
            {
                "left_image": list(cond_file["arm_0_image"].values()),
                "right_image": list(cond_file["arm_1_image"].values()),
                "dimension": list(cond_file["reward_dimension"].values()),
                "left_reward": list(cond_file["arm_0_reward"].values()),
                "right_reward": list(cond_file["arm_1_reward"].values()),
                "max_reward": list(cond_file["max_reward"].values()),
                "min_reward": list(cond_file["min_reward"].values()),
                "choice": beh_file["choices"],
                "reward_received": beh_file["points"],
                "cond_file": beh_file_no,
                "trial": range(TRIAL_NO),
            }
        )

        par_df["bonus_payment"] = float(beh_file["money"]) - BASE_PAY
        par_df["include"] = np.where(beh_file["include"] == "yes", 1, 0)
        par_df["participant"] = id_generator()
        par_df["regret"] = par_df["max_reward"] - par_df["reward_received"]
        par_df["chance_regret"] = (par_df["max_reward"] - par_df["min_reward"]) / 2
        par_df["correct"] = np.where(par_df.regret == 0, 1, 0)
        df_list.append(par_df)

    all_df = pd.concat(df_list)
    filter_chance(all_df, "reward_learning")
    return None


def parse_category_data() -> None:
    """
    Read and parse behavioural csv files of category learning into pandas dataframes.

    Make 2 dataframes:
    1. all_df: includes all participants
    2. keep_df: includes only those with above 50% accuracy
    """
    BASE_PAY = 1.5
    TRIAL_NO = 120

    project_root = get_project_root()

    beh_files = glob.glob(
        join(project_root, "experiments", "category_learning", "data", "task_*.csv")
    )
    df_list = []

    for beh_file in beh_files:
        df = pd.read_csv(beh_file)
        trial_df = df[df.trial_type == "image-keyboard-response"].reset_index(drop=True)

        par_df = pd.DataFrame(
            {
                "image": trial_df.stimulus.to_list(),
                "choice": np.where(trial_df.response == "j", 1, 0),
                "true_category_name": trial_df.trueCategory,
                "true_category_binary": np.where(
                    trial_df.trueCategory == "Julty", 1, 0
                ),
                "correct": trial_df.correct,
                "cond_file": trial_df.cond_file_no,
                "participant": id_generator(),
                "include": np.where(
                    json.loads((df.tail(1)["response"]).values[0])["include"] == "Yes",
                    1,
                    0,
                ),
                "bonus_payment": np.round(float(df.tail(1).current_pay) - BASE_PAY, 2),
                "dimension": (trial_df.cond_file_no - 1) % 3,
                "trial": range(TRIAL_NO),
            }
        )

        df_list.append(par_df)

    all_df = pd.concat(df_list)
    filter_chance(all_df, "category_learning")

    return None


def filter_chance(
    df: pd.DataFrame,  # dataframe to filter
    task: str,  # 'reward_learning' or 'category_learning'
) -> None:
    """
    Take large behavioural df, and make two copies:

    1. Dataframe as is
    2. Dataframe with $p(correct) > .5$ & those who said their data should be included

    Write these on the disk.
    """

    allowed_tasks = ["reward_learning", "category_learning"]
    assert task in allowed_tasks, f"{task} not in {allowed_tasks}"

    project_root = get_project_root()
    df.to_csv(
        join(project_root, "data", "human_behavioural", task, "all.csv"),
        index=False,
    )
    accuracy_df = df.groupby(["participant", "include"], as_index=False)[
        "correct"
    ].mean()
    keep_participants = accuracy_df[
        (accuracy_df.correct > 0.5) & (accuracy_df.include)
    ].participant.to_list()
    keep_df = df[df.participant.isin(keep_participants)].reset_index(drop=True)

    keep_df.to_csv(
        join(project_root, "data", "human_behavioural", task, "above_chance.csv"),
        index=False,
    )
    return None
