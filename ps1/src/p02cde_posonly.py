import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t')
    x_test, t_test = util.load_dataset(test_path, label_col='t')
    clf = LogisticRegression()
    clf.fit(x_train, t_train)
    t_pred = clf.predict(x_test)
    np.savetxt(pred_path_c, t_pred, "%d")
    util.plot(x_test, t_test, clf.theta, pred_path_c+".png")

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y')
    x_test, y_test = util.load_dataset(test_path, label_col='y')
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    np.savetxt(pred_path_d, y_pred, fmt="%d")
    util.plot(x_test, t_test, clf.theta, pred_path_d+".png")

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y')
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_valid_pred = clf.predict_score(x_valid)
    alpha, count = 0, 0
    for i in range(y_valid.shape[0]):
        if y_valid[i]:
            alpha += y_valid_pred[i]
            count += 1
    alpha /= count
    y_pred = clf.predict_score(x_test)
    y_pred = ((y_pred / alpha) >= 0.5).astype(np.int)
    np.savetxt(pred_path_e, y_pred, fmt="%d")
    util.plot(x_test, t_test, clf.theta / alpha, pred_path_e+".png", correction=1 + np.log(2 / alpha - 1) / clf.theta[0])
    # *** END CODER HERE
