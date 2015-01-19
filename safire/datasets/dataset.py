"""Base class for safire datasets."""
import logging
import gensim
import theano
import safire.utils.transcorp


#class DatasetABC(gensim.utils.SaveLoad):
"""Base class for datasets. A dataset is a wrapper around something
that supports slice retrieval and adds the following functionality:

* batch retrieval
* may advertise the space in which the data lives

Retrieval through ``__getitem__`` calls is still possible.

The dataset does *not* define the format of the retrieved data (dense vs.
sparse, numpy vs. gensim...). Use a DataFormatter for that. [NOT
IMPLEMENTED]

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

Datasets are containers: they are fully specified at initialization
and afterwards are only accessed.

Train/dev/test split is handled at the dataset level.
"""
#    pass


class DatasetABC(gensim.utils.SaveLoad):
    """This is the old Dataset, reworked into a wrapper for an IndexedCorpus
    (like, for instance, the ShardedCorpus). It can also serve as a wrapper
    for a Dataset, in order to be subclassable to something that performs
    an on-the-fly transformation of the data."""
    def __init__(self, data, dim=None, test_p=None, devel_p=None):
        """Constructs a Dataset wrapper for the given data.

        The default train-dev-test split is by proportion of data.
        Subclasses may override it.
        
        :type data: gensim.corpora.IndexedCorpus
        :param data: An indexed corpus, or anything that supports
            slice retrieval.

        :type dim: int
        :param dim: Dimension of input space (think number of input
                     neurons).

        :param test_p: Proportion of data to be used as test. Will be taken
            from the end of the dataset.

        :param devel_p: Proportion of data to be used as development set.
            Will be taken from before the test set.
        """
        try:
            x = data[0:1]
        except TypeError:
            #raise TypeError('Dataset initialized with non-sliceable type'
            #                ' ({0})'.format(type(data)))
            logging.warn('Dataset initialized with non-sliceable type'
                         ' ({0})'.format(type(data)))

        if not dim:
            dim = self.derive_dimension(data)

        self.data = data

        self.dim = dim
        self.n_in = dim # Input row dimension/shape
        self.n_out = dim # Input row dimension/shape

        self.test_p = 0.0
        if test_p:
            self.test_p = test_p
        self._test_doc_offset = len(self) - int(len(self) * self.test_p)

        self.devel_p = 0.0
        if devel_p:
            self.devel_p = devel_p
        self._devel_doc_offset = self._test_doc_offset \
                                 - int(len(self) * self.devel_p)

    def n_train_batches(self, batch_size):
        """Determines how many batches of given size the training data will
        be split into.

        :type batch_size: int
        :param batch_size: The intended size of one batch of the data.

        :returns: The number of batches the training data will be split into
            for the given ``batch_size``.
        """
        return self._devel_doc_offset / batch_size

    def n_devel_batches(self, batch_size):
        """Determines how many batches of given size the training data will
        be split into.

        :type batch_size: int
        :param batch_size: The intended size of one batch of the data.

        :returns: The number of batches the training data will be split into
            for the given ``batch_size``.
        """
        return (self._test_doc_offset - self._devel_doc_offset) / batch_size

    def n_test_batches(self, batch_size):
        """Determines how many batches of given size the training data will
        be split into.

        :type batch_size: int
        :param batch_size: The intended size of one batch of the data.

        :returns: The number of batches the training data will be split into
            for the given ``batch_size``.
        """
        return (len(self) - self._test_doc_offset) / batch_size

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

    def _get_batch(self, subset, kind, b_index, b_size,
                   dtype=theano.config.floatX):
        """Retrieves a segment of the data, specified by the arguments.

        :type subset: str
        :param subset: One of ``'train'``, ``'devel'`` or ``'test'``.
            Specifies which subset of the dataset should be used.

        :type kind: str
        :param kind: One of ``'X'`` or ``'y'``. Specifies whether we want
            the inputs or the response.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
            1 for second, etc.)

        :type b_size: int
        :param b_size: Size of one batch.

        :raises: ValueError

        """

        lbound = b_index * b_size

        if subset == 'train':
            if kind == 'X':
                if lbound + b_size > self._devel_doc_offset:
                    raise ValueError('Too high batch index and/or batch size'
                                     ' (%d, %d); training dataset has only %d documents.' % (b_index, b_size, self._devel_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified:'
                                 ' %s (unsupervised datasets only support \'X\')' % kind)

        elif subset == 'devel':
            if kind == 'X':
                lbound += self._devel_doc_offset
                if lbound + b_size > self._test_doc_offset:
                    raise ValueError('Too high batch index and/or batch size'
                                     ' (%d, %d); devel dataset has only %d documents.' % (b_index, b_size, self._test_doc_offset - self._devel_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: '
                                 '%s (unsupervised datasets only support \'X\')' % kind)

        elif subset == 'test':
            if kind == 'X':
                lbound += self._test_doc_offset
                if lbound > len(self):
                    raise ValueError('Too high batch index and/or batch size'
                                     ' (%d, %d); testing dataset has only %d documents.' % (b_index, b_size, len(self) - self._test_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: %s (unsupervised'
                                 ' datasets only support \'X\')' % kind)

        else:
            raise ValueError('Wrong batch subset specified: %s (datasets only supports \'train\', \'devel\', \'test\').' % subset)

    def _build_batch(self, lbound, batch_size, dtype=theano.config.floatX):
        """Given the first index of a batch and batch size, builds the batch
        from the corpus.
        """
        result = self[lbound:lbound+batch_size]
        return result

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def derive_dimension(data):
        """Derives the dimension (space) of the data. This method is
        called during initialization and should be overridden by
        SimpleDataset and CompositeDataset to enforce constraints on
        data structure."""
        return safire.utils.transcorp.dimension(data)


class Dataset(DatasetABC):
    """What should be in the default Dataset?"""
    pass


class SimpleDataset(DatasetABC):
    """This is essentially a renaming of the Dataset class. Indicates
    a Simple/Composite split architecture. Most datasets will be Composite.

    TODO: Refactor Dataset direct __getitem__ behavior into SimpleDataset.
    """
    pass


class CompositeDataset(DatasetABC):
    """Allows combining datasets into more complex spaces. Also allows naming
    datasets (this is useful for train/dev/test splits and features/targets
    splits, as defined by specialization subclasses

    Initialized with a tuple or list of Datasets (or something that can
    act as a tuple, like a list).

    >>> features = DatasetABC([[1], [2], [3]], dim=1)
    >>> targets = DatasetABC([[-1], [-2], [-3]], dim=1)
    >>> composite = CompositeDataset((features, targets), names=('features', 'targets'))
    >>> composite[1:3]
    ([[2], [3]], [[-2], [-3]])
    >>> composite['targets'][:2]
    [[-1], [-2]]
    >>> composite.dim
    (1, 1)

    Can also be recursive:

    >>> recursive = CompositeDataset((composite, composite), names=('first', 'second'))
    >>> recursive.dim
    ((1, 1), (1, 1))
    >>> recursive[1:3]
    (([[2], [3]], [[-2], [-3]]), ([[2], [3]], [[-2], [-3]]))

    """
    def __init__(self, data, dim=None, names=None,
                 test_p=None, devel_p=None):

        # Check lengths
        self.length = len(data[0])
        super(CompositeDataset, self).__init__(data, dim=dim,
                                               test_p=test_p,
                                               devel_p=devel_p)

        for d in data:
            if len(d) != self.length:
                raise ValueError('All composite dataset components must have'
                                 ' the same length.') # TODO: more informative

        if names:
            if len(names) != len(data):
                raise AssertionError('Dataset names too many or too few'
                                     ' ({0}) for {1} component'
                                     ' datasets.'.format(len(names),
                                                         len(data)))
        else:
            names = []
        self.names = names
        self.names_dict = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, item):
        try:
            return tuple(d[item] for d in self.data)
        except TypeError:
            if isinstance(item, str):
                return self.data[self.names_dict[item]]
            else:
                raise

    def __len__(self):
        return self.length

    @staticmethod
    def derive_dimension(data):
        return tuple(d.dim for d in data)


class SupervisedDataset(CompositeDataset):

    def __init__(self, data, test_p=None, devel_p=None):
        super(SupervisedDataset, self).__init__(data,
                                                names=('features', 'targets'),
                                                test_p=test_p,
                                                devel_p=devel_p)


# This only makes sense when implementing some extra batch
# retrieval methods and NOT using __getitem__ directly (would return
# a 1-tuple).
class UnsupervisedDataset(CompositeDataset):

    def __init__(self, data, test_p=None, devel_p=None):
        if len(data) != 1:
            raise ValueError('UnsupervisedDataset is composed of only 1'
                             'component dataset (not {0})'.format(len(data)))
        super(UnsupervisedDataset, self).__init__(data,
                                                  names=['features'],
                                                  test_p=test_p,
                                                  devel_p=devel_p)
