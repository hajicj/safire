#!/usr/bin/env python
"""
``math.py`` is a library that provides mathematical utility functions.
"""
import logging
import numpy
import math

__author__ = 'Jan Hajic jr.'


def precision(prediction, true):
    """Computes the precision of the prediction item predicting members of the
    true item.

    Precision for an empty prediction is defined as 1."""
    t = frozenset(true)
    total = 1.0 * len(prediction)
    if total == 0.0:
        return 1.0
    hits = 1.0 * len([p for p in prediction if p in t])
    prec = hits / total
    return prec


def recall(prediction, true):
    """Computes the precision of the prediction item predicting members of the
    true item.

    Recall for an empty prediction is defined as 0."""
    t = frozenset(true)
    total = 1.0 * len(true)
    if total == 0.0:
        return 1.0
    hits = 1.0 * len([p for p in prediction if p in t])
    rec = hits / total
    return rec


def f_score(prediction, true, w_prec=0.5, w_rec=0.5):
    """Computes the weighed f-score of the prediction item predicting members
    of the test item.

    Currently only supports f1-score."""
    rec = recall(prediction, true)
    prec = precision(prediction, true)
    if rec == 0.0 or prec == 0.0:
        return 0.0
    fsc = ((w_prec + w_rec) * prec * rec) / ((w_rec * rec) + (w_prec * prec))
    return  fsc


def kappa(a1, a2, length=12.0):
    """Computes Cohen's kappa for the given annotation members. The ``length``
    parameter is used to see how many """
    a1_tagged = 1.0 * len(a1)
    p1_tagged = a1_tagged / length
    a1_untagged = 1.0 * (length - len(a1))
    p1_untagged = a1_untagged / length

    a2_tagged = 1.0 * len(a2)
    p2_tagged = a2_tagged / length
    a2_untagged = 1.0 * (length - len(a2))
    p2_untagged = a2_untagged / length

    a1_fset = frozenset(a1)
    hits = 1.0 * len([a for a in a2 if a in a1_fset])
    both_missed = length - (a1_tagged + a2_tagged - hits)

    p_agree = (hits + both_missed) / (2 * length)
    p_random = p1_tagged * p2_tagged + p1_untagged * p2_untagged

    if p_random == 1.0:
        return 0.0

    kappa = (p_agree - p_random) / (1.0 - p_random)
    return kappa

##############################################################################

# Evaluation functions


def avg_pairwise_mutual_precision(items):
    """Computes the average pairwise mutual precision of all the given items."""
    pairwise_mutual_precisions = []
    for i, it_i in enumerate(items[:-1]):
        for j, it_j in enumerate(items[i+1:]):
            pmp = pairwise_mutual_precision(it_i, it_j)
            pairwise_mutual_precisions.append(pmp)
    avg_pmp = avg(pairwise_mutual_precisions)
    return avg_pmp


def pairwise_mutual_precision(i1, i2):
    """Computes the mutual precision of two items. Mutual precision is defined
    as the average of the proportion of members in ``i1`` that are members of
    ``i2`` and vice versa."""
    if len(i1) == 0 or len(i2) == 0:
        return 0.5
    fi1 = frozenset(i1)
    fi2 = frozenset(i2)
    p1 = (1.0 * len([m for m in i1 if m in fi2])) / (1.0 * len(i1))
    p2 = (1.0 * len([m for m in i2 if m in fi1])) / (1.0 * len(i2))
    p_avg = (p1 + p2) / 2.0
    return p_avg


def avg(iterable, weights=None):
    """Returns the average from the iterable (assumes that the iterable can
    be summed). If weights are given, the weighed average is computed.

    Weights do not have to be normalized."""
    if not weights:
        return sum(iterable) / (1.0 * len(iterable))
    else:
        if len(weights) != len(iterable):
            raise ValueError('Iterable and weights lengths do not match! (%d vs. %d)' % (len(iterable), len(weights)))
        wsum = 1.0 * sum(weights)
        normalized_weights = [ (1.0 * w) / (1.0 * wsum) for w in weights ]
        weighed_iterable = [ i * w
                             for i, w in zip(iterable, normalized_weights)]
        return sum(weighed_iterable)


def mse(x, y):

    assert len(x) == len(y), 'X and Y must be of the same length! (x: %d, y: %d)' % (len(x), len(y))

    squared_error = (x - y) ** 2
    N = float(len(x))
    result = sum(squared_error) / N
    return result


def rmse(x, y):
    return numpy.sqrt(mse(x,y))


def rn_rmse(x, y):
    """Range-normalized RMSE."""
    maximum = max(max(x), max(y))
    minimum = min(min(x), min(y))
    if (maximum == minimum):
        logging.warn('Zero range!')
        return 0.0
    rnrmse = rmse(x,y) / (maximum - minimum)
    return rnrmse


def crmse(x, y):
    """Mean-normalized RMSE."""
    return rmse(x,y) / (avg(x+y) / 2)


def maxn_sparse_rmse(x, y):
    """Computes proportion of attainable RMSE. Assumes X and Y are sparse enough
    so that if X is sorted min->max and Y is sorted max->min, one of X[i], Y[i]
    will always be 0. For x,y >= 0, this definitely does maximize RMSE."""
    sorted_x = numpy.sort(x)
    rsorted_y = numpy.sort(y)[::-1]
    max_rmse = rmse(sorted_x, rsorted_y)
    actual_rmse = rmse(x,y)
    if (max_rmse == 0.0):
        logging.debug('All zero feature pair!')
        return numpy.nan
    maxnrmse = actual_rmse / max_rmse
    return maxnrmse


def mae(x, y):
    """Mean absolute error."""
    assert len(x) == len(y), 'X and Y must be of the same length! (x: %d, y: %d)' % (len(x), len(y))
    return sum(numpy.absolute(x - y)) / float(len(x))

##############################################################################


def scale_to_unit_covariance(array):
    """Transform the array so that all columns have the same covariance."""
    covariances = numpy.sum(array ** 2, axis=0) / array.shape[0]
    avg_cov = numpy.average(covariances)
    # This is the covariance we will be scaling to.
    # We want to compute the coefficients for each column.
    coefficients = numpy.sqrt(1/covariances) * numpy.sqrt(avg_cov)
    print coefficients.shape
    scaled_array = array * coefficients
    return scaled_array

##############################################################################

# Functions for evaluating metrics on a matrix column grid.


def generate_grid(matrix, fn, use_cols=None, cols=True):
    """Returns mean squared errors between pairs of matrix columns. Results
    are returned as a grid dict with keys in use_cols."""
    n_cols = len(matrix[0])
    if not use_cols:
        use_cols = range(n_cols)

    mses = {}
    for n_i, i in enumerate(use_cols[:-1]):
        cmses = {}
        ivals = matrix[:, i]
        for j in use_cols[n_i+1:]:
            m = fn(ivals, matrix[:, j])
            if numpy.isnan(m):
                logging.debug('NaN in grid computation at %d:%d' % (i,j))
            else:
                cmses[j] = m
        mses[i] = cmses

    return mses


def avg_grid(grid, use_cols=None, cols=True):
    """Computes average MSE between given matrix columns."""

    s = 0.0
    N = 0.0
    for i in grid:
        v_i = grid[i].values()
        if numpy.isnan(sum(v_i)):
            raise ValueError('NaN in grid for feature %d' % i)
        s += sum(v_i)
        N += len(v_i)

        if numpy.isnan(s):
            raise ValueError('NaN in grid sum at feature %d' % i)
        if numpy.isnan(N):
            raise ValueError('NaN in grid size at feature %d' % i)

    return s / N


def grid2sym_matrix(grid):
    """Unrolls the grid into a 2d ndarray. Assumes that the grid is an upper
    diagonal. Doesn't symmetrize (the rest is left at 0.0)."""
    n_rows = len(grid) + 1
    n_cols = grid_max(grid) + 1
    print 'Grid rows:', n_rows
    print 'Grid max:', n_cols

    m = numpy.zeros((n_rows, n_cols))
    for i in grid:
        g = grid[i]
        for j in g:
            m[i,j] = g[j]
            m[j,i] = g[j]

    return m


def grid_max(grid):
    m = max(max(grid.keys()), max([max(grid[i].keys()) for i in grid]))
    return m

##############################################################################


def zero_masking(X, p, theano_rng):
    """

    :param X: theano.tensor.var.TensorVariable
    :param p:

    :type theano_rng: theano.
    :param theano_rng:

    :return:
    """
    return theano_rng.binomial(size=X.shape, n=1, p=p) * X


def uniform_noise(X, p, theano_rng):
    """Adds uniform noise to the given X."""
    return theano_rng.uniform(X, low=0.0, high=p)

##############################################################################


def splice_csr(target, t_from, t_to, source, s_from, s_to):
    """Efficiently takes a slice from a sparse matrix and inserts it into the
    new matrix. Works in-place.

    :param target: The sparse matrix to which the data will be copied.

    :param t_from: The index of the first row of overwritten region.

    :param t_to: The index one-past of the last row of overwritten region.

    :param source: The sparse matrix from which data will be copied.

    :param s_from: The index of the first row of the source data to copy.

    :param s_to: The index one-past of the last row of the source data to copy.
    """
    # Will need to change all indptrs past the row column.
    pass