import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")


def d(x,y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


def dists_matrix(pts):
    """Compute all distances between points in the array."""
    return np.array([[d(pts[i], pts[j]) for i in range(len(pts))] for j in range(len(pts))])


def random_pts(ct):
    """Generate some random points to use as initial values."""
    return np.random.normal(size=2*ct).reshape(ct, 2)


# Example to test formulas and some code
# ======================================
#
# take four points positioned as a quare in the plane.  Use the distances between them as input distances,
# then apply MDS to randomly initialized points; under gradient descent they need to go into a square
# configuration.

p1 = np.array([0, 0])
p2 = np.array([0, 1])
p3 = np.array([1, 0])
p4 = np.array([1, 1])
pts = np.array([p1, p2, p3, p4])

dist_orig = dists_matrix(pts)
embedded_pts = random_pts(len(pts))


def costs(dist_orig, embedded_pts):
    dist_emb = dists_matrix(embedded_pts)
    dist_diff = dist_orig - dist_emb
    return np.sum(np.power(dist_diff, 2))


def grad(dist_orig, embedded_pts):
    """Compute gradient for the cost function"""
    dist_emb = dists_matrix(embedded_pts)
    dist_emb_sqrt = np.sqrt(dist_emb)
    dist_diff = dist_orig - dist_emb
    grad = np.zeros(embedded_pts.shape)
    for i in range(len(embedded_pts)):
        for j in range(len(embedded_pts)):
            if not i == j:
                delta = (embedded_pts[i] - embedded_pts[j]) * (2 * dist_diff[i, j] / dist_emb_sqrt[i,j])
                grad[i] -= delta
    return grad

grad(dist_orig, embedded_pts)

embedded_pts = random_pts(len(pts))
x = np.copy(embedded_pts)
for _ in range(200):
    x -= 0.01 * grad(dist_orig, x)
    print(costs(dist_orig, x))

plt.scatter(x[:,0], x[:,1])
plt.show()

sess = tf.Session()

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6, 7])

tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])
tile_a = tf.expand_dims(tile_a, 2)
tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1])
tile_b = tf.expand_dims(tile_b, 2)

cartesian_product = tf.concat([tile_a, tile_b], axis=2)

cart = sess.run(cartesian_product)

print(cart.shape)
print(cart)

pts = tf.Variable(initial_value=embedded_pts, name='pts')
pts = tf.placeholder(name='pts')
vinit = tf.variables_initializer([pts])
sess.run(vinit)

pts0 = tf.expand_dims(pts, axis=0)
pts1 = tf.expand_dims(pts, axis=1)
prods = tf.pow(tf.subtract(pts0, pts1), 2)
dists = tf.sqrt(tf.reduce_sum(prods, axis=2))
sess.run(dists)

dists_orig_v = tf.Variable(initial_value=dist_orig)
vinit = tf.variables_initializer([dists_orig_v])
sess.run(vinit)
costs = tf.reduce_sum(tf.pow(tf.subtract(dists_orig_v, dists), 2), axis=[0,1])
sess.run(costs, feed_dict={pts: embedded_pts})

gr = tf.gradients(costs, [pts])
sess.run(gr, feed_dict={pts: embedded_pts})


x = tf.constant([1, 2, 3])
y = tf.constant([2, 2, 3])
z = tf.multiply(x, y)
z.eval(session=sess)