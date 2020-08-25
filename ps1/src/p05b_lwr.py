import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(0.5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    # Get MSE value on the validation set
    MSE = np.linalg.norm(y_pred - y_eval) ** 2 / y_eval.shape[0]
    print(MSE)
    # Plot validation predictions on top of training set
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_eval, y_pred, 'ro')
    # plt.show()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        weight = np.exp(np.linalg.norm(self.x.reshape(1, -1, n) - x.reshape(m, 1, n), axis=2) ** 2 / (-2 * self.tau * self.tau))
        res = np.zeros(m)
        # (x.T W x)^-1 x.T W y
        for i in range(m):
            weight_i = np.diag(weight[i])
            theta = np.linalg.inv(self.x.T.dot(weight_i).dot(self.x)).dot(self.x.T).dot(weight_i).dot(self.y)
            res[i] = np.dot(theta, x[i])
        return res
        # *** END CODE HERE ***
