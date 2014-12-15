"""
This module contains classes that ...
"""
import cPickle
import logging
import numpy
import theano
from safire.datasets.dataset import DatasetTransformer

__author__ = "Jan Hajic jr."


class Word2VecSamplingDatasetTransformer(DatasetTransformer):
    """A DatasetTransformer that samples a word from each document vector
    and returns its word2vec representation instead.
    """
    def __init__(self, w2v_transformer, embeddings_matrix=None,
                 pickle_embeddings_matrix=None, n_samples=1):
        """Initializes the transformer -- specifically the word2vec embeddings.

        :param w2v_transformer: A Word2Vec transformer that provides the
            embeddings dict and id2word mapping. Used during building the
            embeddings matrix; NOT saved. Can be None, if ``embeddings_matrix``
            is provided.

            The id2word mapping of the transformer is expected to map **the
            indices of the dataset on which the transformer will run** to
            words. This is important after a FrequencyBasedTransformer is
            applied.

        :param embeddings_matrix: If there is a saved embeddings matrix, use it.

        :param pickle_embeddings_matrix: If given, pickle the embeddings matrix
            to this file.

        :param n_samples: How many words should be drawn from each document.
        """
        self.embeddings_matrix = None
        self.n_out = None
        if embeddings_matrix:
            self.embeddings_matrix = cPickle.load(embeddings_matrix)
        else:
            self.embeddings_matrix = \
                self.w2v_transformer_to_embedding_matrix(w2v_transformer)

        self.n_in = self.embeddings_matrix.shape[0]   # This is misc.
        self.n_out = self.embeddings_matrix.shape[1]  # This is important.

        if pickle_embeddings_matrix:
            with open(pickle_embeddings_matrix, 'wb') as phandle:
                cPickle.dump(self.embeddings_matrix, phandle, protocol=-1)

        self.n_out = len(self.embeddings[self.embeddings.keys()[0]])

        # Build mapping matrix
        self.embeddngs_matrix = self.build_mapping()
        self.n_samples = 1  # = n_samples  # Currently only supports 1 sample.

        # Pre-compile sampling function
        rng = theano.tensor.shared_randomstreams.RandomStreams(60 * 256)
        pvals = theano.tensor.matrix('pvals', dtype=theano.config.floatX)
        sample = rng.multinomial(pvals=pvals)
        self.sampling_fn = theano.function([pvals], sample)

    def __getitem__(self, batch):
        """Transforms the given batch.

        :param batch: A theano matrix. Expects ``self.n_in`` columns,
            any number of rows (corresponds to batch size during learning).

        :return: The transformed batch. Will have ``self.n_out`` columns.
        """

        # Sample from each document (batch row) one word (column).
        batch_projection = self.get_batch_sample(batch)

        # Use embedding of given word as document vector.
        # - batch_projection is X * n_in,
        # - embeddings_matrix is n_in * n_out
        embeddings = theano.tensor.dot(batch_projection,
                                       self.embeddings_matrix)
        return embeddings

    def get_batch_sample(self, batch):
        """Computes a projection matrix from the batch.

        The projection matrix has frequency (integer) vectors for rows. Each
        batch item (row) is sampled as an unnormalized categorical distribution.
        The number of samples per row is determined by the ``n_samples``
        parameter.

        :type batch: theano.tensor
        :param batch: A batch (theano tensor).

        :return: The projection matrix, also as a theano variable.
        """
        # Normalize by row
        row_sums = theano.tensor.sum(batch, axis=1)
        normalized_batch = theano.tensor.true_div(batch, row_sums)

        # Run through precompiled sampling function
        sample = self.sampling_fn(normalized_batch)
        return sample


    @staticmethod
    def w2v_transformer_to_embedding_matrix(w2v_transformer):
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