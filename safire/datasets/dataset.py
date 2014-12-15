from gensim.interfaces import TransformationABC
from safire.datasets.unsupervised_dataset import UnsupervisedDataset


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


