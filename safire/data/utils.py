#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as TT


def concat_sparse(sparse1, sparse2, dim1, dim2):
    """Concatenates two sparse gensim vectors. The dimension of the first
    will be added to all keys of the second.
    """
    result = sparse1
    concat_sparse2 = [ (s[0] + dim1, s[1]) for s in sparse2 ]
    result.extend(concat_sparse2)

    return result


def as_shared(data, borrow=True):
    """Loads the given data = [features, response] into shared
    variables and returns a tuplet (shared_features, shared_response)
    to mimic the original ``data`` structure.

    Use this function when you wish to convert to shared
    a feature-response subset of a supervised dataset.

    .. warn::

      Assumes that the response variable is discrete.

    :type data: tuple
    :param data: A tuple ``(features, response)`` of anything
                 that can successfully undergo ``numpy.asarray``.

    :type borrow: bool
    :param borrow: Set to ``True`` if the ``theano.shared`` call
                   should use ``borrow=True`` (default), otherwise
                   set to ``False``. See Theano shared variable
                   documentation for what ``borrow`` means.

    :returns: Tuple ``(shared_features, shared_response)`` where
              member are Theano shared variables.
              The function assumes that the response is discrete
              and casts it to ``'int32'`` using ``theano.tensor.cast()``.
    """
    features, response = data
    shared_features = theano.shared(numpy.asarray(features,
                                                  dtype = theano.config.floatX),
                                    borrow=borrow)
    shared_response = theano.shared(numpy.asarray(response,
                                                  dtype = theano.config.floatX),
                                    borrow=borrow)

    return shared_features, TT.cast(shared_response, 'int32')


def as_shared_list(data, borrow=True):
    """Loads the given data = ``[var1, var2...]`` into shared
    variables and returns a list ``[shared_var1, shared_var2...]``
    to mimic the original ``data`` structure.


    :type data: iterable
    :param data: An iterable ``[var1, var2...]`` of anything
                 that can successfully undergo ``numpy.asarray``.

    :type borrow: bool
    :param borrow: Set to ``True`` if the ``theano.shared`` call
                   should use ``borrow=True`` (default), otherwise
                   set to ``False``. See Theano shared variable
                   documentation for what ``borrow`` means.

    :returns: List where all members are Theano shared variables
              constructed from the original iterable.
    """
    shared_data = []
    for subset in data:
        shared_subset = theano.shared(numpy.asarray(subset,
                                                    dtype = theano.config.floatX),
                                      borrow = borrow)
        shared_data.append(shared_subset)

    return shared_data