import numpy as np
import util


from linear_model import LinearModel

def normalize(x):
    zero_mean_x = x - x.mean(axis=0, keepdims=True)
    return zero_mean_x / zero_mean_x.var(axis=0, keepdims=True)

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path)
    x_eval, y_eval = util.load_dataset(eval_path)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    y_pred = clf.predict(x_eval)
    util.plot(x_eval, y_eval, clf.theta, pred_path + ".png")
    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred, "%d", ",")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # Find phi, mu_0, mu_1, and sigma
        pos_cnt = y.sum()
        phi = pos_cnt / m
        mu_0 = x[y == 0].sum(axis=0) / (m - pos_cnt)
        mu_1 = x[y == 1].sum(axis=0) / pos_cnt
        mu = np.stack([mu_0, mu_1])
        zero_mean_x = x - mu[y.astype(np.int)]
        sigma = np.dot(zero_mean_x.T, zero_mean_x) / m
        mu_0, mu_1 = mu_0.reshape(-1, 1), mu_1.reshape(-1, 1)

        self.phi, self.mu_0, self.mu_1, self.sigma = phi, mu_0, mu_1, sigma
        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = np.log(phi / (1 - phi)) + 0.5 * np.dot(mu_0.T, np.dot(sigma_inv, mu_0)) - 0.5 * np.dot(mu_1.T, np.dot(sigma_inv, mu_1))
        self.theta = np.concatenate([theta_0, theta])
        # *** END CODE HERE ***

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x.dot(self.theta))).reshape(-1)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        res = self.sigmoid(x)
        res = (res >= 0.5).astype(np.int)
        return res
        # *** END CODE HERE
