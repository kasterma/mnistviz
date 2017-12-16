import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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