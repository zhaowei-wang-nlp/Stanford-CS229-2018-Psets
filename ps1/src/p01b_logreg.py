import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path)
    x_eval, y_eval = util.load_dataset(eval_path)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_eval, delimiter=',', fmt="%d")
    util.plot(x_eval, y_eval, clf.theta, pred_path+".png")
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        m, n = x.shape

        y = y.reshape(m, 1)
        self.theta = np.zeros((n, 1))
        # theta = theta - l'/l''
        change = 1e10
        
        while change > 1e-5:
            l_one = np.zeros((n, 1))
            l_two = np.zeros((n, n))
            y_hat = self.sigmoid(x)
            l_one += ((y_hat - y) * x).sum(axis=0).reshape(-1, 1) / m
            l_two += np.dot(x.T, x * (y_hat * (1 - y_hat))) / m
                
            revision = -np.dot(np.linalg.inv(l_two), l_one)
            self.theta += revision
            change = np.linalg.norm(revision, ord=1)
        # *** END CODE HERE ***

    def sigmoid(self, x):
        # 1/(1+e^{-theta*x})
        return 1 / (1 + np.exp(- np.dot(x, self.theta)))

    def predict_score(self, x):
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        m, n = x.shape
        res = self.sigmoid(x).reshape(-1)
        return res

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        m, n = x.shape

        res = self.sigmoid(x).reshape(-1)
            
        res = (res >= 0.5).astype(np.int)
        return res
        # *** END CODE HERE ***
