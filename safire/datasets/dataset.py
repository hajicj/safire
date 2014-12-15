"""Base class for safire datasets."""


class Dataset(object):
    """Base storage class for datasets.

    Only expectation: data has a train-devel-test split.
    """
    def __init__(self, data, n_in):
        """Constructs a Dataset from the given data.
        
        :type data: tuple
        :param data: A triplet of data items. No other expectations
                     than a length of 3 are enforced.
                     
        :type n_in: int
        :param n_in: Dimension of input space (think number of input neurons).
        """

        assert len(data) == 3, 'Dataset initialized with incorrect train-devel-test ternary structure.'
        # TODO: assertions about dimensionality?
        # TODO: check for type, so that we're not sharing already shared variables

        self.n_in = n_in # Input row dimension/shape

        self.train = data[0]
        self.devel = data[1]
        self.test = data[2]

    def _get_batch(self, subset, kind, b_index, b_size):
        raise NotImplementedError


