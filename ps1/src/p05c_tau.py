import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression

def plot_lwr(x_train, y_train, x_eval, y_pred, save_path):
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_eval, y_pred, 'ro')
    plt.savefig(save_path)


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    MSE_values = []
    model_list = []
    # Fit a LWR model with the best tau value
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_eval)
        MSE = np.linalg.norm(y_pred - y_eval) ** 2 / y_eval.shape[0]
        print("tau {}, MSE {}".format(tau, MSE))
        MSE_values.append(MSE)
        model_list.append(clf)
        plot_lwr(x_train, y_train, x_eval, y_pred, "output/tau_{}.png".format(tau))

    idx = np.argmin(MSE_values)
    best_model, best_tau = model_list[idx], tau_values[idx]
    y_test_pred = best_model.predict(x_test)
    test_MSE = np.linalg.norm(y_test_pred - y_test) ** 2 / y_test.shape[0]
    print("best tau {}, MSE on the test split {}".format(best_tau, test_MSE))

    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
