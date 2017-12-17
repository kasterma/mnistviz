# MNIST viz

Notes and code for reading
[Visualizing MNIST: An Exploration of Dimensionality Reduction](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)
by Chris Olah.

## Notes

What does the subspace of the 28x28 = 784 dim subspace of greyscale images look like that is relevant for MNIST?

* Submanifold?
* Blobs with tentacles?

Some randomly generated image does not look likely to be close to a digit.

See randomimage for this; but there also checked some properties of random mnist images.  There are 69 pixels
that are zero in every image (hence at least the images live in 784 - 69 dim subspace).  Every image also has 
at least 433 real zero values (and 436 value <= 0.1).

Clearly for human recognition there is no effect if we add a noise image with uniform noise in range [0, 0.1].  This
supports the blobs with tentacles view (any valid image can be manipulated in any direction in the 784 dim subspace
without impacting human recognition much; can imagine difficult to recognize images to be influenced).

If we fix two pixels and look at the generated 2-dim subspace, get some patterns (i.e. there is information there), but
hard to imagine a 2-dim subspace in which strong decisions can be made.

Q1: if we use a decision tree of given size, can we find two pixels where we get reasonably high accuracy?

A1: with first thing that comes to mind we get to 37% accuracy with two pixels.  On the PCA embedding with the same
method we get 47%.  Neither is very good, but they are relatively close to eachother.  Compute to find the two pixels
was rather high though (we did a rather exhaustive search).

Q2: can we find two pixels where the distribution is really close to uniform?  I.e. the 2-dim subspace looks like
generated by taking uniform choices 10 times.  Proper formulation of this?

The plot of classes in the PCA dimensions shows that PCA leads to fair spread of the classes.