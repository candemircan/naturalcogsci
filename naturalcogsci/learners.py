from __future__ import annotations


__all__ = [
    "CategoryLearner",
    "RewardLearner",
]


import numpy as np
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.base import clone


class CategoryLearner:
    def __init__(
        self,
        estimator=LogisticRegression(
            max_iter=4000
        ),  # Linear model to be used for the task. Defaults to `sklearn.linear_model.LogisticRegression`.
    ):
        """
        A class of agent that is used to model the category-learning learning task
        using a linear model of choosing.
        """
        self.estimator = estimator
        # below are place holders
        self.X = np.zeros(1)
        self.y = np.zeros(1)
        self.values = np.zeros(1)
        self.weights = np.zeros(1)
        self.mean = 0
        self.std = 1

    def _predict(self, trial: int):
        """
        Make predictions for the observation for the given trial

        Args:
            trial (int): trial number
        """

        # scale test
        X_test = self.X[trial] - self.mean
        X_test /= self.std
        X_test = X_test.reshape(1, -1)

        self.values[trial, :] = self.estimator.predict_proba(X_test)

    def _learn(self, trial: int):
        """

        Fit the model on observations up until the given trial.
        If that does not include observations belonging to both classes,
        use the pseudo-observations to make predictions

        Args:
            trial (int): trial number
        """

        if 0 in self.y[: trial + 1] and 1 in self.y[: trial + 1]:
            self.estimator = clone(self.estimator)
            train_X = self.X[: trial + 1]

            # update scaling parameters
            self.mean = train_X.mean(axis=0)
            self.std = train_X.std(axis=0)
            self.std = np.where(self.std == 0, 1, self.std)

            train_X -= self.mean
            train_X /= self.std

            self.estimator.fit(train_X, self.y[: trial + 1])

    def fit(self, X: np.ndarray, y: np.ndarray):  # Observations  # Category
        """
        Fit the model to the task in a sequential manner like participants did the task.

        Also save the evolving weights into an array.

        See the structure needed for X and y in the `helpers` module.
        """

        self.X = X
        self.y = y
        self.values = np.zeros((self.X.shape[0], 2))

        # give pseudo-observations so the model can make predictions
        self.estimator.fit(np.zeros((2, self.X.shape[1])), np.array([0, 1]))

        for trial in range(self.X.shape[0]):
            self._predict(trial)
            self._learn(trial)


class RewardLearner:
    def __init__(
        self,
        estimator=BayesianRidge(),  #  Linear model to be used for the task. Defaults to `sklearn.linear_model.BayesianRidge`.
    ):
        """
        A class of agent that is used to model the reward-guided learning task
        using a linear model of choosing.

        """
        self.estimator = estimator
        # below are place holders
        self.X = np.zeros(1)
        self.y = np.zeros(1)
        self.values = np.zeros(1)
        self.weights = np.zeros(1)

    def fit(self, X: np.ndarray, y: np.ndarray):  # Observations  # Reward
        """
        Fit the model to the task in a sequential manner like participants did the task.

        Also save the evolving weights into an array

        See the structure needed for X and y in the `helpers` module.
        """

        self.X, self.y = X, y
        self.values = np.zeros([self.X.shape[0], 2])
        self.weights = np.zeros([self.X.shape[0], self.X.shape[2]])

        # initialise scaling values
        mean = np.zeros(self.X.shape[2])
        std = np.ones(self.X.shape[2])

        for trial in range(self.X.shape[0]):
            test_X = self._get_test_data(trial)

            # scale test data
            test_X -= mean
            test_X /= std
            self._predict(test_X, trial)

            training_X, training_y = self._get_training_data(trial)

            # get scaling parameters for training data
            mean = training_X.mean(axis=0)
            std = training_X.std(axis=0)
            std = np.where(std == 0, 1, std)

            training_X -= mean
            training_X /= std

            self._learn(training_X, training_y)

            self.weights[trial, :] = self.estimator.coef_

    def _predict(self, test_X: np.ndarray, trial: int):
        """
        Make predictions for the giben observations and save them.
        Leave 0 predictions for the first trial for both options.

        Args:
            test_X (np.ndarray): novel observations -> option (right==1) x feature
            trial (int): trial number
        """
        if trial:
            self.values[trial, 0] = self.estimator.predict(test_X[0].reshape(1, -1))
            self.values[trial, 1] = self.estimator.predict(test_X[1].reshape(1, -1))

    def _learn(self, training_X: np.ndarray, training_y: np.ndarray):
        """
        Fit model to given data. Clone the used estimator by detaching the data.

        Args:
            training_X (np.ndarray): observations -> trial (interleaved both options) x feature
            training_y (np.ndarray): rewards -> trial (interleaved both options)
        """
        self.estimator = clone(self.estimator)
        self.estimator.fit(training_X, training_y.ravel())

    def _get_test_data(self, trial: int):
        """
        Collapse 3D observations into 2D by having the two options in an interleaved manner
        in axis 0. Axis 1 is features.

        Args:
            trial (int): trial number

        """

        test_X = np.stack([self.X[trial, 0, :], self.X[trial, 1, :]], axis=0)

        return test_X

    def _get_training_data(self, trial: int):
        """
        Collapse 3D observations into 2D by merging trial (axis 0) and options (axis 2)
        axes into axis 0, where we have options in an interleaved manner.

        Collapse 2D rewards into 1D where trials and options are interleaved in the same
        manner as axis 0.

        Args:
            trial (int): trial number
        """
        training_X = np.concatenate(
            [self.X[: trial + 1, 0, :], self.X[: trial + 1, 1, :]], axis=0
        )
        training_y = np.concatenate(
            [self.y[: trial + 1, 0], self.y[: trial + 1, 1]], axis=0
        )[:, np.newaxis]

        return training_X, training_y