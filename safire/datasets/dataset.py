"""Base class for safire datasets."""


class DatasetABC(object):
    """Base class for datasets. A dataset is a wrapper around something
    that supports slice retrieval and adds the following functionality:

    * batch retrieval
    * may advertise the space in which the data lives

    The dataset does *not* define the format of the retrieved data (dense vs.
    sparse, nump vs. gensim...). Use a DataFormatter for that.

    >>> corpus = [[(1, 22.3), (3, 1.8), (4, 0.97)], [(2, 0.5) (3, 11.6)]]
    >>> dataset = DatasetABC(data=corpus, dim=4)
    >>> dataset.train_X_batch(0, 1)
    [[(1, 22.3), (3, 1.8), (4, 0.97)]]
    >>> dataset.train_X_batch(0, 2)
    [[(1, 22.3), (3, 1.8), (4, 0.97)], [(2, 0.5) (3, 11.6)]]

    Note that the dataset always returns **batches**: even if your batch size
    is just one item, it will be the first member of a single-member iterable
    (by default a list, as seen in the example).

    The data lives in various *spaces*. A space is just a fancy name for the
    structure of a data point. The items in our dataset are four-dimensional
    vectors, so the dimension of the dataset is simply 4. (Constructor argument
    ``dim``.)

    Dimensionality information is very important for conversions between
    different output formats (dense vs. sparse) and for models.

    Supervised vs. Unsupervised datasets
    ------------------------------------

    A simple dataset is unsupervised. All "columns" have the same role. In this
    perspective, making a dataset supervised is the matter of the *model*: the
    model should interpret certain columns as input variables and certain
    columns as output variables. However, to set up a model, the dimension of
    the input and output must be known.

    !!!BRAINSTORMING!!!
    ^^^^^^^^^^^^^^^^^^^

    TODO: design a solution
    - Split: initialization time vs. runtime
    - Init time: read dimensionality, determine inputs structure?
    - Runtime: feed data to inputs (as: model_handle.train(*inputs!)

    Question: do I have to know at dataset build time which vars are input
    and which are response? (SupervisionTransformer/SupervisedDataset?)

    Case study with Word2VecSamplingTransformer: when will the sampling be
    applied? What about a *supervised* setting with the sampler?

    Principle: **There should only be a dataset when feeding data into a model.
    No intermediate datasets, everything that actually works with the data
    is a corpus/transformer.** A Dataset is an interface into the model.

    Typically, the dataset will work with a TransformedCorpus... but we need
    indexing: what about a TransformedIndexedCorpus that subclasses
    TransformedCorpus and adds __getitem__ that just passes the output of
    a __getitem__ call on the underlying corpus through the ``obj``?
    (Make a utility function that can construct this indexable TransformedCorpus
    if both the ``corpus`` and the ``obj`` member support slicing calls?)

    In the end: we need a **reshaper** that can map from one space to another.

    What next:
    ^^^^^^^^^^

    0. Write a basic wrapper that implements the old Dataset functionality.
    1. Get a basic pipeline to work: only create a simple dataset that can feed
       batches for an unsupervised model.
    2. Decide on support for supervised datasets. May coincide heavily with
       refactoring the cost function outside the models.

    Internals
    ---------

    The default implementation assumes that there is an underlying object
    that supports slicing operation on ``__getitem__`` call and can return its
    ``__len__`` that gets passed as the ``data`` parameter to the constructor.

    Datasets are stateless. There should be no need for persistence.
    Train/dev/test split is handled at the dataset level.
    """


class Dataset(object):
    """This is the old Dataset."""
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

