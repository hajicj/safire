#!/usr/bin/env python

import theano

from safire.data.utils import as_shared
from safire.datasets.dataset import Dataset


class SupervisedDataset(Dataset):
    """Storage class for supervised datasets.

    Expects the data to be split into a train/devel/test set, with
    a response vector for each of the datasets.

    Supports exporting batches.
    """
    def __init__(self, data, n_in, n_out):
        """Constructs a :class:`SupervisedDataset` from the given data.

        :type data: tuple
        :param data: A triplet of train-devel-test sets. Each of the sets must
                     be a tuple of ``(input, response)`` where ``input`` are
                     the data points and ``response`` are the gold-standard
                     labels. Expects each of the ``input``s to be of the same
                     dimension as supplied in ``n_in``.

        :type n_in: int
        :param n_in: Dimension of input space (the space where the data points
                     live).

        :type n_out: int
        :param n_out: How many different labels there are in the data. (Think
                      number of neurons in classification output layer.)
        """

        assert len(data) == 3, 'Dataset initialized with incorrect train-devel-test ternary structure.'
        # TODO: assertions about dimensionality?
        # TODO: check for type, so that we're not sharing already shared variables

        self.n_in = n_in # Input row dimension/shape
        self.n_out = n_out # Output dimension - number of classes, NOT
                               # dimension of output space
                               # (think: how many output neurons?)

        self.train_X = None
        self.train_y = None
        self.devel_X = None
        self.devel_y = None
        self.test_X = None
        self.test_y = None

        if (isinstance(data[0][0], theano.tensor.sharedvar.TensorSharedVariable)):
            self.train_X, self.train_y = data[0]
        else:
            assert len(data[0][0]) == len(data[0][1]), 'Dataset-train: unequal length of train features and response.'
            self.train_X, self.train_y = as_shared(data[0])

        if (isinstance(data[1][0], theano.tensor.sharedvar.TensorSharedVariable)):
            self.devel_X, self.devel_y = data[1]
        else:
            assert len(data[1][0]) == len(data[1][1]), 'Dataset-train: unequal length of train features and response.'
            self.devel_X, self.devel_y = as_shared(data[1])

        if (isinstance(data[2][0], theano.tensor.sharedvar.TensorSharedVariable)):
            self.test_X, self.test_y = data[2]
        else:
            assert len(data[2][0]) == len(data[2][1]), 'Dataset-train: unequal length of train features and response.'
            self.test_X, self.test_y = as_shared(data[2])

    def n_train_batches(self, batch_size):
        """Determines how many batches of given size the training data will be
        split into.

        :type batch_size: int
        :param batch_size: The intended size of one batch of the data.

        :returns: The number of batches the training data will be split
                  into for given batch_size.
        """
        return self.train_X.get_value(borrow=True).shape[0] / batch_size

    def n_devel_batches(self, batch_size):
        """Determines how many batches of given size the development data will
        be split into.

        :type batch_size: int
        :param batch_size: The intended size of one batch of the data.

        :returns: The number of batches the devel data will be split
                  into for given batch_size.
        """
        return self.devel_X.get_value(borrow=True).shape[0] / batch_size


    def n_test_batches(self, batch_size):
        """Determines how many batches of given size the test data will
        be split into.

        :type batch_size: int
        :param batch_size: The intended size of one batch of the data.

        :returns: The number of batches the test data will be split
                  into for given batch_size.
        """
        return self.test_X.get_value(borrow=True).shape[0] / batch_size

    # Batch retrieval

    def train_X_batch(self, b_index, b_size):
        """Slices a batch of ``train_X`` for given batch index and batch size.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable ``train_X`` starting at
                  ``b_index * b_size`` and ending at ``(b_index + 1) *
                  b_size``.

        :raises: ValueError
        """
        return self._get_batch('train', 'X', b_index, b_size)

    def train_y_batch(self, b_index, b_size):
        """Slices a batch of ``train_y`` for given batch index and batch size.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable ``train_y`` starting at
                  ``b_index * b_size`` and ending at ``(b_index + 1) *
                  b_size``.

        :raises: ValueError
        """
        return self._get_batch('train', 'y', b_index, b_size)

    def devel_X_batch(self, b_index, b_size):
        """Slices a batch of ``devel_X`` for given batch index and batch size.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable ``devel_X`` starting at
                  ``b_index * b_size`` and ending at ``(b_index + 1) *
                  b_size``.

        :raises: ValueError
        """
        return self._get_batch('devel', 'X', b_index, b_size)

    def devel_y_batch(self, b_index, b_size):
        """Slices a batch of ``devel_y`` for given batch index and batch size.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable ``devel_y`` starting at
                  ``b_index * b_size`` and ending at ``(b_index + 1) *
                  b_size``.

        :raises: ValueError
        """
        return self._get_batch('devel', 'y', b_index, b_size)

    def test_X_batch(self, b_index, b_size):
        """Slices a batch of ``test_X`` for given batch index and batch size.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable ``test_X`` starting at
                  ``b_index * b_size`` and ending at ``(b_index + 1) *
                  b_size``.

        :raises: ValueError
        """
        return self._get_batch('test', 'X', b_index, b_size)

    def test_y_batch(self, b_index, b_size):
        """Slices a batch of ``test_y`` for given batch index and batch size.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable ``test_y`` starting at
                  ``b_index * b_size`` and ending at ``(b_index + 1) *
                  b_size``.

        :raises: ValueError
        """
        return self._get_batch('test', 'y', b_index, b_size)

    def _get_batch(self, subset, kind, b_index, b_size):
        """Retrieves a segment of the data, specified by the arguments.

        :type subset: str
        :param subset: One of ``'train'``, ``'devel'`` or ``'test'``.
                       Specifies which subset of the dataset should be used.

        :type kind: str
        :param kind: One of ``'X'`` or ``'y'``. Specifies whether we want
                     the inputs or the response.

        :type b_index: int

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
                        1 for second, etc.)

        :type b_size: int
        :param b_size: The size of one batch.

        :returns: A slice of the shared variable specified by the ``subset``
                  and ``kind`` arguments. The slice begins at ``b_index *
                  b_size`` and ends at ``(b_index + 1) * b_size``.

        :raises: ValueError

        """

        lbound = b_index * b_size
        rbound = (b_index + 1) * b_size

        if (subset == 'train'):
            if (kind == 'X'):
                return self.train_X[lbound:rbound]
            elif (kind == 'y'):
                return self.train_y[lbound:rbound]
            else:
                raise ValueError('Wrong batch kind specified: %s (use \'X\' or \'y\')' % kind )
        elif (subset == 'devel'):
            if (kind == 'X'):
                return self.devel_X[lbound:rbound]
            elif (kind == 'y'):
                return self.devel_y[lbound:rbound]
            else:
                raise ValueError('Wrong batch kind specified: %s (use \'X\' or \'y\')' % kind )
        elif (subset == 'test'):
            if (kind == 'X'):
                return self.test_X[lbound:rbound]
            elif (kind == 'y'):
                return self.test_y[lbound:rbound]
            else:
                raise ValueError('Wrong batch kind specified: %s (use \'X\' or \'y\')' % kind )
        else:
            raise ValueError('Wrong subset specified: %s (use \'train\', \'devel\' or \'test\')' % subset )
