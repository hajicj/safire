"""Base class(es) for datasets. A dataset is a wrapper around something
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

import logging
import gensim
from gensim.interfaces import TransformationABC
import theano
import safire.datasets
import safire.utils
import safire.utils.transcorp


# Doesn't DatasetABC implement the IndexedCorpus interface?
# Should it be made explicit?
# TODO: Try to implement DatasetABC as a CorpusABC subclass.
#       This would greatly simplify a lot of transcorp.py functions.
class DatasetABC(gensim.utils.SaveLoad):
    """This is the old Dataset, reworked into a wrapper for an IndexedCorpus
    (like, for instance, the ShardedCorpus). It can also serve as a wrapper
    for a Dataset, in order to be subclassable to something that performs
    an on-the-fly transformation of the data."""
    def __init__(self, data, dim=None, test_p=None, devel_p=None,
                 ensure_dense=True):
        """Constructs a Dataset wrapper for the given data.

        The default train-dev-test split is by proportion of data.
        Subclasses may override it.
        
        :type data: gensim.corpora.IndexedCorpus, numpy.ndarray
        :param data: An indexed corpus, or anything that supports
            slice retrieval.

        :type dim: int
        :param dim: Dimension of input space (think number of input
                     neurons).

        :param test_p: Proportion of data to be used as test. Will be taken
            from the end of the dataset.

        :param devel_p: Proportion of data to be used as development set.
            Will be taken from before the test set.

        :param ensure_dense: If set, makes sure that the ``data`` member really
            outputs dense numpy ndarrays. This is the expected behavior for
            Datasets, although in principle there is nothign wrong with
            outputting gensim sparse vectors (and guaranteeing a dimension at
            the same time).
        """
        try:
            logging.debug('Dataset init: inspecting sliceability...')
            x = data[0:2]
        except TypeError:
            logging.warn('Dataset initialized with non-sliceable type'
                         ' ({0})'.format(type(data)))

        if not dim:
            dim = self.derive_dimension(data)

        _data = data
        if ensure_dense:
            logging.info('Ensuring dense output...')
            _data = safire.utils.transcorp.convert_to_dense(_data)
        self.data = _data

        self.dim = dim
        self.n_in = dim   # Input row dimension/shape
        self.n_out = dim  # Input row dimension/shape

        self.set_test_p(test_p)
        self.set_devel_p(devel_p)

    def set_test_p(self, test_p=None):
        """Helper function for setting a proportion of data as test data. For
        simple use cases only; use a CompositeDataset with a train/dev and
        test split for more principled experiments."""
        self.test_p = 0.0
        if test_p:
            self.test_p = test_p
        self._test_doc_offset = len(self) - int(len(self) * self.test_p)
        # Devel proportion is counted from the bottom end of the test data
        # proportion, so if we increased test_p, we would get overlapping
        # test data and devel data!
        if hasattr(self, 'devel_p'):
            self.set_devel_p(self.devel_p)

    def set_devel_p(self, devel_p):
        """Helper function for setting a proportion of data as heldout data. For
        simple use cases only; use a CompositeDataset with a train/dev and
        test split for more principled experiments."""
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
                                     ' ({0}, {1}); training dataset has only '
                                     '%{2} documents.'
                                     ''.format(b_index,
                                               b_size,
                                               self._devel_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: {0} (simple'
                                 ' datasets only support \'X\')'.format(kind))

        elif subset == 'devel':
            if kind == 'X':
                lbound += self._devel_doc_offset
                if lbound + b_size > self._test_doc_offset:
                    raise ValueError('Too high batch index and/or batch size'
                                     ' ({0}, {1}); devel dataset has only {2}'
                                     ' documents.'
                                     ''.format(b_index,
                                               b_size,
                                               self._test_doc_offset -
                                                    self._devel_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: '
                                 '{0} (simple datasets only support'
                                 ' \'X\')'.format(kind))

        elif subset == 'test':
            if kind == 'X':
                lbound += self._test_doc_offset
                if lbound + b_size > len(self):
                    raise ValueError('Too high batch index and/or batch size'
                                     ' ({0}, {1}); testing dataset has only {2}'
                                     ' documents.'
                                     ''.format(b_index,
                                               b_size,
                                               len(self) - self._test_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: {0} (unsupervised'
                                 ' datasets only support \'X\')'.foramat(kind))

        else:
            raise ValueError('Wrong batch subset specified: {0} (datasets '
                             'only supports \'train\', \'devel\','
                             ' \'test\').'.format(subset))

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

    def __iter__(self):
        """Implemented in order to be compatible with the CorpusABC
        interface."""
        for i in xrange(len(self)):
            yield self[i]

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

    However, it only currently supports building this tree-like structure one
    by one. Trying ``composite = CompositeDataset(((data1, data2), data3))``
    will fail.
    """
    def __init__(self, data, dim=None, names=None,
                 test_p=None, devel_p=None, aligned=True):
        """Initializes a CompositeDataset.

        :param data:

        :param dim:

        :param names:

        :param test_p:

        :param devel_p:

        :type aligned: bool
        :param aligned: If set, will expect that all the individual datasets
            from ``data`` have the same length. If unset, will not check this
            and advertise the length of the first given dataset as its length;
            only do this if you are flattening the dataset immediately after
            initialization!

        """
        self.aligned = aligned
        # Check lengths
        self.length = len(data[0])  # TODO: This is very temporary.
        super(CompositeDataset, self).__init__(data, dim=dim,
                                               test_p=test_p,
                                               devel_p=devel_p,
                                               ensure_dense=False)
        # The composite dataset doesn't care if input or output are dense or
        # not...

        if self.aligned:
            for d in data:
                if len(d) != self.length:
                    raise ValueError('All composite dataset components must '
                                     'have the same length. (Lengths: '
                                     '{0}) Are you sure the CompositeDataset'
                                     'should be aligned?'
                                     ''.format(tuple((len(d) for d in data))
                    ))

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
        """Retrieval from a composite dataset has several modes:

        >>> features = DatasetABC([[1], [2], [3]], dim=1)
        >>> targets = DatasetABC([[-1], [-2], [-3]], dim=1)
        >>> composite = CompositeDataset((features, targets), names=('features', 'targets'))
        >>> composite[1:3]
        ([[2], [3]], [[-2], [-3]])
        >>> composite.__getitem__((1, 2))
        ([2], [-3])

        """
        try:
            # For retrieving a different index from each data point
            if isinstance(item, tuple):
                return tuple([d[item[i]] for i, d in enumerate(self.data)])
            else:
                return tuple([d[item] for d in self.data])
        except (TypeError, IndexError):
            if isinstance(item, str):
                return self.data[self.names_dict[item]]
            else:
                raise

    def __len__(self):
        # Ugly hack - returns a structure instead of a number... doesn't work
        # with test_p and devel_p, though, so I disabled it temporarily.
        #if not self.aligned:
        #    return tuple([len(d) for d in self.data])
        return self.length

    @staticmethod
    def derive_dimension(data):
        return tuple(d.dim for d in data)


class SupervisedDataset(CompositeDataset):
    """This dataset supports training supervised models, through combining
    a ``features`` and ``targets`` dataset.

    Under the hood, it re-routes ``train_X_batch()`` and related methods to
    access the 'features' and 'targets' component datasets, so that you can use
    the SupervisedDataset "transparently" without having to take the
    CompositeDataset mechanism into account.
    """
    def __init__(self, data, test_p=None, devel_p=None):
        super(SupervisedDataset, self).__init__(data,
                                                names=('features', 'targets'),
                                                test_p=test_p,
                                                devel_p=devel_p)
        # Shortcut pointers
        self.features = self.data[self.names_dict['features']]
        self.targets = self.data[self.names_dict['targets']]

    # TODO: copy over train_Y_batch et al.
    def _get_batch(self, subset, kind, b_index, b_size,
                   dtype=theano.config.floatX):
        """Retrieves a segment of the data, specified by the arguments.

        :type subset: str
        :param subset: One of ``'train'``, ``'devel'`` or ``'test'``.
            Specifies which subset of the dataset should be used.

        :type kind: str
        :param kind: One of ``'X'`` or ``'y'``. Specifies whether we want
            the features or the targets.

        :type b_index: int
        :param b_index: The order of the batch in the dataset (0 for first,
            1 for second, etc.)

        :type b_size: int
        :param b_size: Size of one batch.

        :raises: ValueError

        :returns: Whatever the given dataset will return as a batch. (Typically
            a numpy.ndarray.)

        """
        if kind == 'X':
            return self.features._get_batch(subset=subset,
                                            kind='X',
                                            b_index=b_index,
                                            b_size=b_size,
                                            dtype=dtype)
        elif kind == 'y':
            return self.targets._get_batch(subset=subset,
                                           kind='X',
                                           b_index=b_index,
                                           b_size=b_size,
                                           dtype=dtype)

        else:
            raise ValueError('Wrong batch kind specified: {0} (supervised'
                             ' datasets only support \'X\' and \'y\')'
                             ''.format(kind))

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


# This only makes sense when implementing some extra batch
# retrieval methods and NOT using __getitem__ directly (would return
# a 1-tuple).
class UnsupervisedDataset(CompositeDataset):

    def __init__(self, data, test_p=None, devel_p=None):
        if len(data) != 1:
            raise ValueError('UnsupervisedDataset is composed of only 1'
                             'component dataset (not {0})'.format(len(data)))
        super(UnsupervisedDataset, self).__init__(data,
                                                  names=tuple(['features']),
                                                  test_p=test_p,
                                                  devel_p=devel_p)


class TransformedDataset(Dataset):
    """A class that enables stacking dataset transformations in the training
    stage, similar to how corpus transformations are done during preprocessing.

    Instead of overriding ``__iter__`` like TransformedCorpus, will need to
    override ``_get_batch``.

    (Note: shouldn't this also implement __iter__, to be a TransformedCorpus as
    well? Goes together with the question whether Datasets shouldn't also be
    IndexedTransformedCorpora...)
    """
    def __init__(self, obj, dataset):
        """Initializes the transformer.

        :type obj: DatasetTransformer
        :param obj: The transformer through which a batch from the ``dataset``
            is run.

        :type dataset: Dataset
        :param dataset: The underlying dataset whose _get_batch calls are
            overridden.

        """
        self.data = dataset
        self.obj = obj

        self.n_out = self.obj.n_out
        self.dim = safire.utils.transcorp.dimension(obj)
        # This is the preferred interface.

        # This is a naming hack, because the model setup() method expects
        # model.n_in == data.n_in. Note that this doesn't matter much: this
        # is a dataset, so it only has ONE dimension. (The transformer, on the
        # other hand, does have a separate input and output dimension.)
        self.n_in = self.n_out

    def _get_batch(self, subset, kind, b_index, b_size):
        """The underlying mechanism for retrieving batches from
        the dataset. Note that this method is "hidden" -- the learner and other
        classes that utilize a Dataset call methods ``train_X_batch()`` etc.,
        but all these methods internally "redirect" to ``_get_batch()``.

        In this class, the batch is retrieved from the underlying Dataset
        and then transformed through the transformer's ``__getitem__`` method.

        :param subset:

        :param kind:

        :param b_index:

        :param b_size:

        :return:
        """
        batch = self.data._get_batch(subset=subset,
                                     kind=kind,
                                     b_index=b_index,
                                     b_size=b_size)
        transformed_batch = self.obj[batch]
        return transformed_batch

    def __getitem__(self, item):

        # print 'Calling __getitem__ on TransformedDataset with obj of type {0}' \
        #       'and item {1}'.format(type(self.obj), item)
        data = self.data[item]
        # print '  TransformedDataset.__getitem__: operating on type {0} with ' \
        #       'shape {1}; item {2}'.format(type(data), data.shape, item)
        result = self.obj[data]
        # print '      Result: type {0} with shape {1}'.format(type(result),
        #                                                      result.shape)
        return result


class DatasetTransformer(TransformationABC):
    """DatasetTransformer is a base class analogous to gensim's
    :class:`TransformationABC`, but it operates efficiently on theano batches
    instead of gensim sparse vectors.

    Constraints on dataset transformations:

    * Batch size cannot change (one item for one item).
    * Dimension may change; must provide ``n_out`` attribute.
       * This means output dimension must be a fixed number known at
         transformer initialization time (but can be derived from initializaton
         parameters, or can be a parameter directly).
    * The ``__getitem__`` method must take batches (matrices, incl. theano
      shared vars!)

    """
    def __init__(self):
        self.n_out = None
        self.n_in = None

    def __getitem__(self, batch):
        """Transforms a given batch.

        :type batch: numpy.array, theano.shared
        :param batch: A batch.

        :return: The transformed batch. If a dataset is given instead of
            a batch, applies the transformer and returns a TransformedDataset.
        """
        if isinstance(batch, safire.datasets.dataset.Dataset):
            return self._apply(batch)

        raise NotImplementedError

    def _apply(self, dataset, chunksize=None):

        if not isinstance(dataset, safire.datasets.dataset.Dataset):
            raise TypeError('Dataset argument given to _apply method not a'
                            'dataset (type %s)' % type(dataset))

        transformed_dataset = TransformedDataset(dataset=dataset, obj=self)
        return transformed_dataset