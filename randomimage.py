# Generate a random image in mnist image space

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random_image = np.random.uniform(0, 1, 28 * 28).reshape(28, 28)

plt.imshow(random_image, cmap=plt.cm.Greys)
plt.show()

# random image from test set

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

plt.imshow(mnist.train.images[15].reshape(28, 28), cmap=plt.cm.Greys)
plt.show()

images_df = pd.DataFrame(mnist.train.images)
images_df.iloc[np.random.choice()].value_counts()

# Look at the zero values in these images
# Note: we are looking at *real* zero values, not approx zero.

column_zeros = images_df.apply(lambda col: np.sum(col == 0))
np.sum(column_zeros == 55000)
# 69; i.e. there are 69 pixels in which there is a zero in every picture

hist_zeros = images_df.apply(lambda row: np.sum(row == 0), axis=1).value_counts()
hist_zeros.index.min()
# 433; i.e. every picture has at least 433 zero.

# Now for approx zero
hist_near_zeros = images_df.apply(lambda row: np.sum(row < 0.1), axis=1).value_counts()
hist_near_zeros.index.min()


def min_leq_bound(bd):
    hist = images_df.apply(lambda row: np.sum(row <= bd), axis=1).value_counts()
    return hist.index.min()


# quick sanity check
assert(min_leq_bound(1.0) == 784)
min_leq_bound(0.1)
# 436: only 3 higher then the number of real zeros

noise_image = np.random.uniform(0, 0.1, 28 * 28).reshape(28, 28)

plt.imshow(mnist.train.images[15].reshape(28, 28) + noise_image, cmap=plt.cm.Greys)
plt.show()