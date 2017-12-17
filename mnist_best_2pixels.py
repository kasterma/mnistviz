# Find the two pixels on which a decision tree gets the best results

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, GridSearchCV

from tensorflow.examples.tutorials.mnist import input_data

import time

mnist = input_data.read_data_sets("MNIST_data/")

# Code setup; work out on example what code should look like

idx_1 = 330
idx_2 = 600

dtc = DecisionTreeClassifier()

xs = mnist.train.images[:, idx_1]
ys = mnist.train.images[:, idx_2]

dtc.fit(np.array([xs, ys]).T, mnist.train.labels)
preds = dtc.predict(np.array([xs, ys]).T)
accuracy_score(preds, mnist.train.labels)

# Loop going over all options; we'll use the parameters that worked for PCA and then after we have
# a good option run a grid search for better ones (and then maybe iterate)

scores = np.zeros([28*28, 28*28])

start = time.monotonic()
for idx_1 in range(784):
    print(f"idx_1 {idx_1} {time.monotonic() - start}")
    for idx_2 in range(784):
        xs = mnist.train.images[:, idx_1]
        ys = mnist.train.images[:, idx_2]
        X_train, X_test, lab_train, lab_test =\
            train_test_split(np.array([xs, ys]).T, mnist.train.labels)

        dtc = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_split=200)
        dtc.fit(X_train, lab_train)

        scores[idx_1, idx_2] = dtc.score(X_test, lab_test)

# full run took 9352 seconds

np.save("scores.npy", scores)

scores.max()
# 37%; with the parameters from the pca result.  Pretty close already.
np.unravel_index(np.argmax(scores), scores.shape)

# the following locations have quite decent results; so there try a grid search for better values.
candidates = np.where(scores > 0.35)

grid = {"max_depth": range(7, 15),
        "min_samples_split": [10, 50, 100, 200, 300, 400],
        "criterion": ["gini", "entropy"]}

search_results = []
for idx in range(len(candidates[0])):
    idx_1 = candidates[0][idx]
    idx_2 = candidates[1][idx]
    dtc = DecisionTreeClassifier()
    gdtc = GridSearchCV(dtc, param_grid=grid)

    xs = mnist.train.images[:, idx_1]
    ys = mnist.train.images[:, idx_2]
    X_train, X_test, lab_train, lab_test =\
        train_test_split(np.array([xs, ys]).T, mnist.train.labels)
    gdtc.fit(X_train, lab_train)
    score = gdtc.score(X_test, lab_test)
    print(f"result: {score}")
    print(f"params: {gdtc.best_params_}")
    search_results.append({'idx_1': idx_1, 'idx_2': idx_2, 'score': score, 'params': gdtc.best_params_})

idx_max = np.argmax([x['score'] for x in search_results])
max_res = search_results[idx_max]

idx_1 = max_res['idx_1']
idx_2 = max_res['idx_2']
dtc = DecisionTreeClassifier(**max_res['params'])

xs = mnist.train.images[:, idx_1]
ys = mnist.train.images[:, idx_2]
dtc.fit(np.array([xs, ys]).T, mnist.train.labels)
score = dtc.score(np.array([mnist.test.images[:, idx_1], mnist.test.images[:, idx_2]]).T,
                  mnist.test.labels)
# 37%