import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

# Focus on two pixels (indicated by idx_1 and idx_2) and visualize the 2-dim subspace obtained

idx_1 = 400
idx_2 = 500

xs = mnist.train.images[:, idx_1]
ys = mnist.train.images[:, idx_2]

plt.scatter(xs, ys, c=mnist.train.labels+1, cmap=plt.cm.RdYlGn)
plt.show()
# reasonable spread, but certainly not uniform

idx_1 = 330
idx_2 = 600
# nice spread, but not uniform

idx_1 = 1
idx_2 = 2
# single point at origin; these two pixels are always zero

pca = PCA(n_components=2)
pca.fit(mnist.train.images)

plt.imshow(pca.components_[1].reshape([28,28]))
plt.show()

plt.imshow(pca.components_[0].reshape([28,28]))
plt.show()

pca.transform(mnist.train.images[0].reshape(1, -1))

decomp = pca.transform(mnist.train.images)
plt.scatter(decomp[:, 0], decomp[:, 1], c=mnist.train.labels+1)
plt.show()

decomp.shape

dtc = DecisionTreeClassifier()
dtc.fit(decomp, mnist.train.labels)
preds = dtc.predict(decomp)
accuracy_score(preds, mnist.train.labels)

decomp_test = pca.transform(mnist.test.images)
preds_test = dtc.predict(decomp_test)
accuracy_score(preds_test, mnist.test.labels)

np.min(cross_validate(dtc, decomp, mnist.train.labels, cv=5)['test_score'])

grid = [{"max_depth": range(7, 11), "min_samples_split": [50, 100, 200, 1000], "criterion": ["gini"]},
{"max_depth": range(7, 11), "min_samples_split": [50, 100, 200, 1000], "criterion": ["entropy"]}]
gdtc = GridSearchCV(dtc, param_grid=grid, verbose=2)
gdtc.fit(decomp, mnist.train.labels)
gdtc.score(decomp, mnist.train.labels)
gdtc.best_params_

gdtc.score(decomp_test, mnist.test.labels)

# so with the PCA 2-dim we can quickly get to an accuracy of 47%; lets see if we can find 2 pixels that can do the same