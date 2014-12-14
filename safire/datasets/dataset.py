import cPickle
from gensim.interfaces import TransformationABC
import numpy
import theano
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

        :return:
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


class Word2VecSamplingDatasetTransformer(DatasetTransformer):
    """A DatasetTransformer that samples a word from each document vector
    and returns its word2vec representation instead.
    """
    def __init__(self, w2v_transformer, embeddings_matrix=None,
                 pickle_embeddings_matrix=None):
        """Initializes the transformer -- specifically the word2vec embeddings.

        :param w2v_transformer: A Word2Vec transformer that provides the
            embeddings dict and id2word mapping. Used during building the
            embeddings matrix; NOT saved. Can be None, if ``embeddings_matrix``
            is provided.

            The id2word mapping of the transformer is expected to map **the
            indices of the dataset on which the transformer will run** to
            words. This is important after a FrequencyBasedTransformer is
            applied.
        """
        self.embeddings_matrix = None
        self.n_out = None
        if embeddings_matrix:
            self.embeddings_matrix = cPickle.load(embeddings_matrix)
        else:
            self.embeddings_matrix = self.build_matrix(w2v_transformer)

        self.n_out = self.embeddings_matrix.shape[1]

        if pickle_embeddings_matrix:
            with open(pickle_embeddings_matrix, 'wb') as phandle:
                cPickle.dump(self.embeddings_matrix, phandle, protocol=-1)

        self.n_out = len(self.embeddings[self.embeddings.keys()[0]])

        # Build mapping matrix
        self.embeddngs_matrix = self.build_mapping()

    def __getitem__(self, batch):
        """Transforms the given batch.

        :param batch:
        :return:
        """

        # Sample from each document (batch row) one word (column).
        batch_projection = self.get_batch_sample(batch)

        # Use embedding of given word as document vector.
        # - batch_projection is X * n_in,
        # - embeddings_matrix is n_in * n_out
        embeddings = theano.tensor.dot(batch_projection,
                                       self.embeddings_matrix)
        return embeddings

    def build_matrix(self, w2v_transformer):
        """Uses the information in a w2v_transformer to build an embeddings
        matrix that speeds up ``__getitem__`` computation.

        :type w2v_transformer: safire.data.word2vec_transformer.Word2VecTransformer
        :param w2v_transformer: A Word2Vec transformer that provides the
            embeddings dict and id2word mapping. Used during building the
            embeddings matrix; NOT saved. Can be None, if ``embeddings_matrix``
            is provided.

            The id2word mapping of the transformer is expected to map **the
            indices of the dataset on which the transformer will run** to
            words. This is important after a FrequencyBasedTransformer is
            applied.

        :return: A theano shared variable with the embeddings matrix.
            The matrix dimensions are ``n_in * n_out``, where ``n_in`` is the
            input document vector dimension and ``n_out`` is the embedding
            dimension.

            .. warning::

                We assume the id2word mapping is dense: all positions are
                used. Furthermore, we assume that embedding with ID ``x``
                corresponds to the ``x``-th column of an input batch.
        """
        n_in = len(w2v_transformer.id2word)
        n_out = w2v_transformer.n_out
        matrix = numpy.zeros((n_in, n_out))

        id2word = w2v_transformer.id2word

        for wid in xrange(len(id2word)):
            embedding = w2v_transformer[(wid, 1)]
            matrix[wid, :] = embedding

        return matrix
