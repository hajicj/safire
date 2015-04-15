#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implements the TransformationABC interface, so that SAFIRE
neural network models can be plugged into a gensim-style
pipeline.
"""
import logging
import cPickle

import gensim.utils
import gensim.matutils
from gensim.interfaces import TransformationABC, TransformedCorpus
from gensim.similarities import Similarity
import numpy
import theano
import theano.printing
import safire
from safire.datasets.dataset import DatasetABC

from safire.learning.interfaces import ModelHandle
from safire.utils import profile_run, gensim2ndarray, IndexedTransformedCorpus
from safire.utils.transcorp import smart_apply_transcorp


class SafireTransformer(TransformationABC):
    """Wraps a SAFIRE model into a gensim-style transformation object.

    Initialized with a Model Handle, Dataset and Learner:

    >>> dataset = loader.load()
    >>> model_handle = MultilayerPerceptron.setup(dataset, ...)
    >>> learner = BaseSGDLearner(n_epochs=3, b_size=100)
    >>> transformer = SafireTransformer(model_handle, learner, dataset)

    *This initialization will run the training,* in line with other gensim
    transformers (models) that train on initialization.

    If you want to load a handle with an already trained model, initialize
    the transformer without a Learner and Dataset:

    >>> model_handle = MultilayerPerceptron.setup(dataset, ...)
    >>> model_handle.save('multilayerperceptron.mhandle')
    >>> loaded_model_handle = ModelHandle.load('multilayerperceptron.mhandle')
    >>> transformer = SafireTransformer(loaded_model_handle)

    This is much faster and the preferred way of doing this, especially for
    runtime initialization.

    Computing outputs
    -----------------

    The items passed to the transformer through ``__getitem__`` will be
    passed as inputs to the neural network inside the model handle, as
    inputs to the ``run`` theano compiled function. The outputs of ``run``
    will be collected and returned again as a corpus.

    Sparse vs. dense output
    -----------------------

    The SafireTransformer accepts both gensim vectors and numpy ndarrays as
    inputs and can produce outputs of either type. The type of throughput is
    controlled by the ``dense_throughput`` attribute. If set, it will expect
    dense input and produce dense output; if unset, it will expect gensim
    input and produce gensim output.

    """
    def __init__(self, model_handle, dataset=None, learner=None,
                 eps=1e-09, chunksize=None, attempt_resume=False,
                 profile_training=False, dense_throughput=False):
        """Initializes the transformer. If necessary, performs model training
        (when the ``dataset`` and ``learner`` arguments are given).

        :type model_handle: safire.learning.interfaces.ModelHandle
        :param model_handle: A :class:`ModelHandle`.

        :type dataset: safire.data.dataset.Dataset
        :param dataset: A Safire :class:`Dataset` that should be used for
            training the model. If not supplied, no training is performed.
            **This is currently inconsistent with gensim: we want a dataset,
            gensim uses corpora** to feed data to model training.

        :type learner: safire.learning.learners.base_sgd_learner.BaseSGDLearner
        :param learner: A :class:`BaseSGDLearner` that will be used to train
            the model. If not supplied, no training is performed.

        :type eps: float
        :param eps: A threshold under which output values will be considered
            noise and not included in the sparse output.

            .. warning::

                Currently not implemented, the gensim builtin ``eps`` of 1e-09
                is used.

        :type chunksize: int
        :param chunksize: The batch size by which input documents will be fed
            to the transformer. It is faster to chunk a larger corpus than to
            transform it one-by-one, because the conversion machinery involved
            in feeding the input to the underlying Theano model and getting it
            back is much less efficient than the mathematical operations on the
            input matrix.

        :type attempt_resume: bool
        :param attempt_resume: If set, the learner will attempt to resume
            training an earlier model.

        :type profile_training: bool
        :param profile_training: If set, will profile the learner run.

        :type dense_throughput: bool
        :param dense_throughput: If set, will (a) assume numpy ndarrays on input
            and (b) not convert output to sparse vectors.
        """

        self.model_handle = model_handle

        if dataset is not None and learner is not None:
            logging.info('Training SAFIRE model...')
            if profile_training:
                s, _ = profile_run(learner.run, self.model_handle, dataset,
                                   resume=attempt_resume)
                print 'Profiling training:'
                print s.getvalue()
            else:
                learner.run(self.model_handle, dataset, resume=attempt_resume)

        # Shortcuts to dimension checking
        self.n_in = self.model_handle.n_in
        self.n_out = self.model_handle.n_out

        self.chunksize = chunksize
        self.eps = eps   # Using this is not implemented.
        self.dense_throughput = dense_throughput

    def save(self, fname, protocol=-1):
        """Saves the transformer. Saving is achieved by getting the handle
        pickleable object and adding the other instance attributes, then
        pickling this dict."""

        pickleable_obj = self._export_pickleable_obj()

        with open(fname, 'wb') as pickle_handle:
            cPickle.dump(pickleable_obj, pickle_handle, protocol=protocol)

    def _export_pickleable_obj(self):
        """
        Exports a dicitonary that can directly be pickled to sufficiently
        describe the transformer.
        """
        init_args = {
            'handle': self.model_handle._export_pickleable_obj(),
            'handle_class': self.model_handle.__class__, # Generality...
            'chunksize': self.chunksize,
            'eps': self.eps
        }

        return init_args

    def __getitem__(self, bow, chunksize=None):

        # We want to do the _apply thing only when the corpus
        # is a stream-style yield()-ing corpus, not when it's
        # a list of sparse vectors. What _apply does is it goes
        # through the corpus in chunks, by setting the chunksize
        # parameter of TransformedCorpus.
        if isinstance(bow, gensim.interfaces.CorpusABC):
            return self._apply(bow, self.chunksize)
        if isinstance(bow, DatasetABC):
            return self._apply(bow, self.chunksize)

        # Convert chunk to dense. We can use the fact that the array of
        # sparse vectors obtained from a corpus is itself a corpus.
        # Both input dense dimensions are known.
        #
        # Note that the ``corpus2dense`` function returns a numpy ndarray with
        # documents as *columns*, while the safire model expects documents as
        # *rows*.

        # Because the model handle needs a 2d input to its run() method,
        # if only a single document is given, we'll need to convert it to
        # 2d. However, before outputting the result (which is also given as
        # a 2d matrix), we should strip the extra dimension, to keep
        # dimensionality of output consistent with dimensionality of input.
        was_single_doc = False

        # logging.debug('SFtrans bag of words type: {0}'.format(type(bow)))
        if not isinstance(bow, numpy.ndarray):

            if self.dense_throughput:
                raise TypeError('dense_throughput set, expecting numpy.ndarray'
                                'input (got: {0})'.format(type(bow)))

            is_corpus, bow = gensim.utils.is_corpus(bow)
            # if not is_corpus:  # If we got a single item: make a one-item
            #                    # corpus from it, to simplify code path.
            #     pass
            if isinstance(bow[0], tuple):  # If we get a single gensim-style
                bow = [bow]                # vector, we convert it to a 1-doc
                was_single_doc = True      # corpus (the handle needs a matrix).

            dense_bow = gensim2ndarray(bow, self.n_in, len(bow))
            # Why not gensim.matutils.corpus2dense? Tansposition!
            # (Due to gensim's corpus2dense returning documents as columns.)

        else:
            if len(bow.shape) == 1:
                was_single_doc = True
            dense_bow = numpy.atleast_2d(bow)

        # Run the model on the dense representation of input.
        dense_out = self.model_handle.run(dense_bow)

        if self.dense_throughput:
            out = dense_out

        else:
            # This is a bad solution if we *want* dense output.
            #logging.debug('Dense_out: {0}'.format(dense_out))
            sparse_out = gensim.matutils.Dense2Corpus(dense_out,
                                                      documents_columns=False)#,
            #eps=self.eps) Param
            # not available in gensim
            # 0.10.1

            sparse_out = list(sparse_out)  # Runs through Dense2Corpus.__iter__
            out = sparse_out

        if was_single_doc:
            out = out[0]

        # if self.n_out == 200:  # Very bad debugging practice
        #     logging.debug('SafireTransformer output: {0}'.format(out))
        #     logging.debug('                  length: {0}'.format(len(out)))
        #     print 'SafireTransformer output: {0}'.format(out)
        #     print '                  length: {0}'.format(len(out))

        return out

    def _apply(self, corpus, chunksize=None):
        """

        :param corpus:
        :param chunksize:
        :return:
        """
        return smart_apply_transcorp(self, corpus, chunksize=chunksize)
        # try:
        #     return IndexedTransformedCorpus(self, corpus, chunksize)
        # except TypeError:
        #     return TransformedCorpus(self, corpus, chunksize)

    @classmethod
    def load(cls, fname):
        """Loads a SafireTransformer pickle dump created with the ``save()``
        mehtod of a SafireTransformer instance."""

        with open(fname, 'rb') as unpickle_handle:
            pickled_obj = cPickle.load(unpickle_handle)

        handle_cls = pickled_obj['handle_class']
        handle = handle_cls._load_from_save_dict(pickled_obj['handle'])
        transformer = cls(handle)

        return transformer