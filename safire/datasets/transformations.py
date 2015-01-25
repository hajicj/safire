import argparse
import logging
from gensim.interfaces import TransformationABC, TransformedCorpus
from gensim.utils import is_corpus
import numpy
from safire.datasets.dataset import Dataset, DatasetABC
from safire.datasets.unsupervised_dataset import UnsupervisedDataset


# TODO: refactor to work with new DatasetABC
from safire.utils import flatten_composite_item


class TransformedDataset(UnsupervisedDataset):
    """A class that enables stacking dataset transformations in the training
    stage, similar to how corpus transformations are done during preprocessing.

    Instead of overriding ``__iter__`` like TransformedCorpus, will need to
    override ``_get_batch``.
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
        self.dataset = dataset
        self.obj = obj

        self.n_out = self.obj.n_out

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
        batch = self.dataset._get_batch(subset=subset,
                                        kind=kind,
                                        b_index=b_index,
                                        b_size=b_size)
        transformed_batch = self.obj[batch]
        return transformed_batch


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
        if isinstance(batch, Dataset):
            return self._apply(batch)

        raise NotImplementedError

    def _apply(self, dataset, chunksize=None):

        if not isinstance(dataset, Dataset):
            raise TypeError('Dataset argument given to _apply method not a'
                            'dataset (type %s)' % type(dataset))

        transformed_dataset = TransformedDataset(dataset=dataset, obj=self)
        return transformed_dataset


class FlattenComposite(DatasetTransformer):
    """This class flattens a composite dataset into a simple dataset. This
    allows on-the-fly integration of various data sources into unstructured
    batches.

    >>> source1 = DatasetABC([[1], [2], [3]], dim=1)
    >>> source2 = DatasetABC([[-1], [-2], [-3]], dim=1)
    >>> composite = CompositeDataset((source1, source2), names=('source1', 'source2'))
    >>> flatten = FlattenComposite(composite)
    >>> flat = flatten[composite]
    >>> flat[1:3]
    array([[ 2, -2],
           [ 3, -3]])

    There are several options of how to flatten a composite dataset:

    * One-by-one: assumes the datasets are aligned. Most restrictive but
      fastest.
    * Pairing: most general -- assumes nothing, gets an iterable of ``dim``-
      shaped indices and outputs one dataset item per index. (For example,
      to flatten a composite dataset consisting of two simple datasets, use
      indices (x, y).)

    To flatten the dataset using pairing data, you need to supply the pairings
    at transformer initialization time.

    >>> indexes = [(1, 2), (2, 1), (2, 2), (0, 2), (0, 1)]
    >>> flatten_indexed = FlattenComposite(composite, indexes)
    >>> flat = flatten_indexed[composite]
    >>> flat[1:3]
    array([[ 3, -2],
           [ -3, 2]])

    Note that when using the indexed approach with data serialized using
    ShardedCorpus, it may significantly slow operations down, as a lot of
    shard-jumping may be involved.

    TODO: If that is the case, you can serialize
    the flat dataset using the SerializationTransformer, which allows
    serializing AND retaining the whole pipeline.

    The idea is that you
    load the entire pipeline, in order to have at your disposal everything
    that you used to create a dataset, but you have already serialized your
    data at some advanced point during the pipeline and will not be going
    backwards from that point. This should allow sufficient introspection
    (i.e. going back to the original text corpus for a dictionary...)
    while retaining efficiency (preprocessing steps are not repeated).
    The SerializationTransformer "swaps out" the incoming ``corpus`` for
    a corpus given at initialization time.
    """
    def __init__(self, composite, indexes=None):
        """Does nothing, only initializes self.composite and self.indexes.
        Everything happens on-the-fly in __getitem__.

        Future: incorporate a ``precompute`` and/or ``preserialize`` flag."""
        self.composite = composite
        self.indexes = indexes

    def _apply(self, dataset, chunksize=None):
        return FlattenedDataset(self, dataset)

    def __getitem__(self, item):

        iscorpus, _ = is_corpus(item)

        if iscorpus or isinstance(item, DatasetABC):
            return self._apply(item)
        else:
            raise ValueError('Cannot apply serializer to individual documents.')


class FlattenedDataset(TransformedCorpus):
    def __init__(self, flatten, dataset):

        self.obj = flatten
        self.indexes = flatten.indexes
        self.corpus = dataset

    def __getitem__(self, item):
        """This is where on-the-fly construction of the flattened dataset
        happens. (Note: will be cached.)

        :param item: An index or slice.

        :return: numpy.ndarray
        """
        if self.indexes is None:
            retrieved = self.corpus[item]
        else:
            retrieved = self.corpus[self.indexes[item]]
        output = self.item2flat(retrieved)
        return output

    @staticmethod
    def item2flat(item):
        """Flattens a (recursive) tuple of numpy ndarrays/scipy sparse matrices
        and stacks them next to each other (i.e.: rows stay rows, columns
        change).

    >>> x = numpy.array([[1], [2], [3], [4]])
    >>> y = numpy.array([[-1], [-3], [-5], [-7]])
    >>> z = numpy.array([[10, 20], [11, 21], [12, 22], [13, 23]])
    >>> item = (x, (y, z))
    >>> FlattenComposite.item2flat(item)
    [1,2,3]

        """
        flattened = list(flatten_composite_item(item))
        stacked = numpy.hstack(flattened)
        return stacked