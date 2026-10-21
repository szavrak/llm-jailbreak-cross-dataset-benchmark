"""TF-IDF + SVM and Logistic Regression baselines."""

import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def run_svm(X_train, X_test, y_train, y_test, seed):
    """Train LinearSVC on TF-IDF features.

    Returns (y_pred, train_time, infer_time, num_params).
    """
    clf = LinearSVC(random_state=seed, max_iter=10000)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    infer_time = time.time() - t0

    # num_params: coef_ shape = (n_classes, n_features) + intercept
    num_params = clf.coef_.size + clf.intercept_.size

    return y_pred, train_time, infer_time, num_params


def run_lr(X_train, X_test, y_train, y_test, seed):
    """Train Logistic Regression on TF-IDF features.

    Returns (y_pred, train_time, infer_time, num_params).
    """
    clf = LogisticRegression(random_state=seed, max_iter=10000, solver='lbfgs')

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    infer_time = time.time() - t0

    num_params = clf.coef_.size + clf.intercept_.size

    return y_pred, train_time, infer_time, num_params
