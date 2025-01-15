import argparse
import tqdm
from os.path import join, isfile

import pandas as pd
from sklearn.linear_model import BayesianRidge, ARDRegression, LogisticRegression
from sklearn.decomposition import PCA

from naturalcogsci.helpers import get_project_root, prepare_training
from naturalcogsci.learners import RewardLearner, CategoryLearner


def fit_regulariser(penalty_type, X, y):
    best_score = 0
    best_alpha = 1
    for alpha in [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        1.5,
        2,
        3,
        4,
        5,
        10,
        15,
        20,
    ]:
        category_learner = CategoryLearner(LogisticRegression(penalty=penalty_type, C=alpha, max_iter=5000, solver="liblinear"))
        category_learner.fit(X, y)
        if category_learner.estimator.score(X, y) > best_score:
            best_score = category_learner.estimator.score(X, y)
            best_alpha = alpha

    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", "-e")
    parser.add_argument("--features", "-f")
    parser.add_argument("--transform", "-t")
    parser.add_argument("--regularisation", "-r")

    args = parser.parse_args()
    experiment = args.experiment
    features = args.features
    transform = args.transform
    regularisation = args.regularisation

    features = features.replace("/", "_")
    print(experiment, features, transform, regularisation, flush=True)

    project_root = get_project_root()
    save_file_name = join(
        project_root,
        "data",
        "learner_behavioural",
        experiment,
        f"{features}_{regularisation}_{transform}.csv",
    )

    if isfile(save_file_name):
        print("file already exists!")
        exit()
    df = pd.read_csv(join(project_root, "data", "human_behavioural", experiment, "above_chance.csv"))

    participants = df.participant.unique()

    model_dfs = []
    for participant in tqdm.tqdm(participants):
        cond_file = df[df.participant == participant]["cond_file"].unique()[0]
        X, y = prepare_training(experiment, features, cond_file)

        N_FEATURES = 49  # this is the number of features in the task
        N_TRIALS = 60
        N_OPTIONS = 2

        if experiment == "reward_learning":
            estimator = BayesianRidge() if regularisation == "l2" else ARDRegression()
            learner = RewardLearner(estimator=estimator)
            if transform == "pca":
                X = X.reshape(N_TRIALS * N_OPTIONS, -1)
                X = PCA(n_components=N_FEATURES).fit_transform(X)
                X = X.reshape(N_TRIALS, N_OPTIONS, -1)

            learner.fit(X, y)

        else:
            penalty_coef = fit_regulariser(regularisation, X, y)
            learner = CategoryLearner(
                estimator=LogisticRegression(
                    penalty=regularisation,
                    C=penalty_coef,
                    max_iter=5000,
                    solver="liblinear",
                )
            )

            X = PCA(n_components=N_FEATURES).fit_transform(X) if transform == "pca" else X

            learner.fit(X, y)

        model_df = df[df.participant == participant].reset_index(drop=True)
        model_df["left_value"] = learner.values[:, 0]
        model_df["right_value"] = learner.values[:, 1]
        model_df["features"] = features
        model_df["transform"] = transform
        model_df["penalty"] = regularisation

        model_dfs.append(model_df)

    large_model_df = pd.concat(model_dfs)
    large_model_df.to_csv(
        save_file_name,
        index=False,
    )

    print("Done!", flush=True)
