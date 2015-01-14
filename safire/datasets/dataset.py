"""Base class for safire datasets."""


class Dataset(object):
    """Base storage class for datasets. A dataset is a wrapper around something
    that supports slice retrieval and adds the following functionality:

    * batch retrieval
    * train-dev-test split

    The dataset does *not* defibne the format of the retrieved data (dense vs.
    sparse, nump vs. gensim...).

    >>> corpus = [[(1, 22.3), (3, 1.8), (4, 0.97)], [(2, 0.5) (3, 11.6)]]
    >>> dataset = Dataset(data=corpus, dim=4)
    >>>

    The default implementation assumes that there is an underlying object
    that supports slicing operation on ``__getitem__`` call and can return its
    ``__len__`` that gets passed as the ``data`` parameter to the constructor.


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


class CompositeDataset(Dataset):
    """Derived class for multiple datasets. On each batch request, returns
    recursively a tuple of batches, one from each dataset it contains.
    If its components are also composite, returns a tuple of tuples, etc.
    (The structure of the data space is determined by the composition structure
    of the data, in math-speak.)
    """
    pass