#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import cPickle
from gensim.corpora import MmCorpus
import numpy
import theano

from gensim import corpora
from gensim import matutils
import time
from safire.utils import profile_run

from .supervised_dataset import SupervisedDataset
from .unsupervised_dataset import UnsupervisedDataset
from .vtextcorpus import VTextCorpus
from .imagenetcorpus import ImagenetCorpus

logger = logging.getLogger(__name__)


class UnsupervisedCorpusDataset(UnsupervisedDataset):
    """This dataset implements a memory-efficient way of feeding
    large corpora by batch. Only the requested batch is loaded
    into memory.

    Internally keeps a reference to a gensim-style indexed corpus,
    which it uses to retrieve batches.
    """

    def __init__(self, mm_corpus_filename, dim, test_p=0.1, devel_p=0.1,
                 serializer=MmCorpus, try_loading=True, save_dataset=None,
                 load_dataset=None, profile_loading=False):
        """Initializes the dataset - loads corpus, sets devel and test offsets.
        Retrieving a batch from the dataset means reading the batch from the
        corpus and converting it to a dense representation.

        :type mm_corpus_filename: str
        :param mm_corpus_filename: Where to find the *.mmcorp file, previously
            created by ``gensim.corpora.MmCorpus.serialize(fname, corpus)``.

        :type dim: int
        :param dim: The desired output *y* dimension of the batches - the size
            of the word vector; number of output columns. This should fit with
            the dimension of the corpus dictionary. (Future plan: implement
            option to truncate to first ``dim`` words.)

        :type test_p: float
        :param test_p: The proportion of documents in the corpus
            used for the test set. The documents will be taken from the end
            of the corpus.

        :type devel_p: float
        :param devel_p: The proportion of documents in the corpus
            used for the development set. The documents will be taken
            from between the training and test sets.

        :type serializer: class froom gensim.interfaces.CorpusABC
        :param serializer: The corpus class that should be used to
            de-serialize the corpus data. Has to be the same as the one
            used to build the data.

        :type try_loading: bool
        :param try_loading: If set, will attempt to load the whole dataset
            into memory on initialization, leading to faster retrieval times.

        :type save_dataset: str
        :param save_dataset: If the dataset is successfully pre-loaded, attempts
            to pickle the numpy ndarrray into this file.

        :type load_dataset: str
        :param load_dataset: If given, will attempt to load a cPickled numpy
            ndarray from the given file.
        """
        self.serializer = serializer

        self.data = serializer(mm_corpus_filename)

        self.dim = dim

        self.n_in = dim # For interface compatibility with models, etc.
        self.n_out = None

        # Compute devel and test set first document, number of batches
        # available.
        self.n_docs = len(self.data)

        self.test_p = test_p
        self._test_doc_offset = self.n_docs - int(self.n_docs * self.test_p)

        self.devel_p = devel_p
        self._devel_doc_offset = self._test_doc_offset - int(self.n_docs * self.devel_p)

        # Various pre-loading
        self.try_loading = try_loading
        self.__loaded_dataset = None

        if load_dataset is not None:
            try:
                with open(load_dataset, 'rb') as unpickle_handle:
                    self.__loaded_dataset = cPickle.load(unpickle_handle)
            except Exception:
                logging.warn('Loading dataset directly from pickled file failed, falling back to slower methods.')

        if self.try_loading:
            if self.__loaded_dataset is not None:
                logging.info('Dataset already loaded, skipping try_loading...')
            else:
                try:
                    if profile_loading:
                        s, self.__loaded_dataset = profile_run(self._load_dataset)
                        print "Loading profiling:"
                        print s.getvalue()
                    else:
                        self.__loaded_dataset = self._load_dataset()
                except MemoryError:
                    logging.warn('Attempted loading failed due to MemoryError.')

        if save_dataset is not None:
            if self.__loaded_dataset is not None:
                logging.info('Pickling dataset ndarray to %s' % save_dataset)
                with open(save_dataset, 'wb') as pickle_handle:
                    cPickle.dump(self.__loaded_dataset, pickle_handle, -1)
            else:
                logging.warn('Attempted loading failed; cannot save dataset.')




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
                    raise ValueError('Too high batch index and/or batch size (%d, %d); training dataset has only %d documents.' % (b_index, b_size, self._devel_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: %s (unsupervised datasets only support \'X\')' % kind)

        elif subset == 'devel':
            if kind == 'X':
                lbound += self._devel_doc_offset
                if lbound + b_size > self._test_doc_offset:
                    raise ValueError('Too high batch index and/or batch size (%d, %d); devel dataset has only %d documents.' % (b_index, b_size, self._test_doc_offset - self._devel_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: %s (unsupervised datasets only support \'X\')' % kind)

        elif subset == 'test':
            if kind == 'X':
                lbound += self._test_doc_offset
                if lbound > len(self):
                    raise ValueError('Too high batch index and/or batch size (%d, %d); testing dataset has only %d documents.' % (b_index, b_size, len(self) - self._test_doc_offset))
                batch = self._build_batch(lbound, b_size, dtype)
                return batch
            else:
                raise ValueError('Wrong batch kind specified: %s (unsupervised datasets only support \'X\')' % kind)

        else:
            raise ValueError('Wrong batch subset specified: %s (datasets only supports \'train\', \'devel\', \'test\').' % subset)

    def _build_batch(self, lbound, batch_size, dtype=theano.config.floatX):
        """Given the first index of a batch and batch size, builds the batch
        from the corpus.
        """
        result = None
        if self.__loaded_dataset is not None:
            result = self.__loaded_dataset[lbound:lbound+batch_size]
        else:
            result = numpy.empty((batch_size, self.dim), dtype=dtype)
            for idx, docno in enumerate(xrange(lbound, lbound + batch_size)):
                doc = self[docno]
                result[idx, :] = doc

        return result.astype(dtype)

    def _shared_batch(self, batch, borrow=True):
        """Given a numpy array, returns it as a Theano shared variable.
        """
        data = numpy.asarray(batch)
        return theano.shared(data, dtype=theano.config.floatX, borrow=borrow)

    def _load_dataset(self, dtype=theano.config.floatX):
        """Attempts to load the dataset into memory for more efficient
         retrieval."""
        logging.info('...attempting to load dataset...')
        start_time = time.clock()
        dataset = numpy.zeros((len(self), self.dim), dtype=dtype)
        for i, doc in enumerate(self.data):
            doc = dict(doc)
            dataset[i][list(doc)] = list(iter(doc.itervalues()))
            if i % 500 == 0:
                logging.info('   at doc %i / %i, so far %d s' % (i,
                                len(self.data),
                                time.clock() - start_time))

        end_time = time.clock()
        total_time = end_time - start_time
        logging.info('   Loading took %d s, %d / doc' % (total_time, total_time / len(self)))
        return dataset

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        """There should be dense output."""
        doc = self.data[idx]
        return matutils.sparse2full(doc, self.dim)
        #return self.indexed_corpus_X[idx]

    def get_sparse(self, idx):
        """Retrieves the given item in the sparse form from the source
        corpus."""
        doc = self.data[idx]
        return doc

    def iter_sparse(self, idx):
        """Iterates through the corpus inside, yielding the sparse vectors."""
        for i in xrange(len(self)):
            doc = self.get_sparse(i)
            return doc

    def __iter__(self):
        """Iterate through the corpus inside, yielding dense data items."""
        for i in xrange(len(self)):
            vector = self[i]
            yield vector


class UnsupervisedVTextCorpusDataset(UnsupervisedCorpusDataset):
    """Adds VTextCorpus capabilities to UnsupervisedCorpusDataset:
    in addition to loading a MmCorpus-serialized corpus and making
    it available as a dataset, it will load a VTextCorpus used to
    build the MmCorpus-serialized data. The VTextCorpus is then
    made available as the ``vtcorp`` member of the object.
    """

    def __init__(self, mm_corpus_filename, vt_corpus_filename,
                 dim=None, test_p=0.1, devel_p=0.1, serializer=MmCorpus,
                 try_loading=True, save_dataset=None,
                 load_dataset=None, profile_loading=False):
        """Initializes the dataset - loads corpora, sets devel and test offset.
        Retrieving a batch from the dataset means reading the batch from the
        corpus and converting it to a dense representation.

        Also loads the specified VTextCorpus and uses it to determine the
        output dimension (as the Dictionary size).

        :type mm_corpus_filename: str
        :param mm_corpus_filename: Where to find the *.mmcorp file, previously
            created by ``gensim.corpora.MmCorpus.serialize(fname, corpus)``
            where ``corpus`` is a VTextCorpus.

        :type vt_corpus_filename: str
        :param vt_corpus_filename: Where to find the *.vtcorp file, created
            through ``corpus.save('vt_corpus_filename')``. The ``corpus`` is
            the VTextCorpus serialized through MmCorpus to create the
            ``mm_corpus_filename`` file.

        :type dim: int
        :param dim: The desired output *y* dimension of the batches - the size
            of the word vector; number of output columns. This should fit with
            the dimension of the corpus dictionary. (Future plan: implement
            option to truncate to first ``dim`` words.)

        :type test_p: float
        :param test_p: The proportion of documents in the corpus
            used for the test set. The documents will be taken from the end
            of the corpus.

        :type devel_p: float
        :param devel_p: The proportion of documents in the corpus
            used for the development set. The documents will be taken
            from between the training and test sets.

        :type serializer: class froom gensim.interfaces.CorpusABC
        :param serializer: The corpus class that should be used to
            de-serialize the corpus data. Has to be the same as the one
            used to build the data.

        :type try_loading: bool
        :param try_loading: If set, will attempt to load the whole dataset
            into memory on initialization, leading to faster retrieval times.

        :type save_dataset: str
        :param save_dataset: If the dataset is successfully pre-loaded, attempts
            to pickle the numpy ndarrray into this file.

        :type load_dataset: str
        :param load_dataset: If given, will attempt to load a cPickled numpy
            ndarray from the given file.

        """
        self.vtcorp = VTextCorpus.load(vt_corpus_filename)

        if not dim:
            dim = len(self.vtcorp.dictionary)
            logging.debug('Setting text dataset dimension to %d' % dim)

        super(UnsupervisedVTextCorpusDataset, self).__init__(mm_corpus_filename,
                 dim, test_p, devel_p, serializer, try_loading, save_dataset,
                 load_dataset, profile_loading)


class UnsupervisedImagenetCorpusDataset(UnsupervisedCorpusDataset):
    """Adds ImgnetCorpus capabilities to UnsupervisedCorpusDataset:
    in addition to loading a MmCorpus-serialized corpus and making
    it available as a dataset, it will load an ImgnetCorpus used to
    build the MmCorpus-serialized data. The ImgnetCorpus is then
    made available as the ``icorp`` member of the object.
    """

    def __init__(self, mm_corpus_filename, ic_corpus_filename,
                 dim=None, test_p=0.1, devel_p=0.1, serializer=MmCorpus,
                 try_loading=True, save_dataset=None,
                 load_dataset=None, profile_loading=False):
        """Initializes the dataset - loads corpora, sets devel and test offset.
        Retrieving a batch from the dataset means reading the batch from the
        corpus and converting it to a dense representation.

        Also loads the specified VTextCorpus and uses it to determine the
        output dimension (as the Dictionary size).

        :type mm_corpus_filename: str
        :param mm_corpus_filename: Where to find the *.mmcorp file, previously
            created by ``gensim.corpora.MmCorpus.serialize(fname, corpus)``
            where ``corpus`` is an ImgnetCorpus.

        :type ic_corpus_filename: str
        :param ic_corpus_filename: Where to find the *.icorp file, created
            through ``corpus.save('ic_corpus_filename')``. The ``corpus`` is
            the ImgnetCorpus serialized through MmCorpus to create the
            ``mm_corpus_filename`` file.

        :type dim: int
        :param dim: The desired output *y* dimension of the batches - the size
            of the word vector; number of output columns. This should fit with
            the dimension of the corpus dictionary. (Future plan: implement
            option to truncate to first ``dim`` words.)

        :type test_p: float
        :param test_p: The proportion of documents in the corpus
            used for the test set. The documents will be taken from the end
            of the corpus.

        :type devel_p: float
        :param devel_p: The proportion of documents in the corpus
            used for the development set. The documents will be taken
            from between the training and test sets.

        :type serializer: class froom gensim.interfaces.CorpusABC
        :param serializer: The corpus class that should be used to
            de-serialize the corpus data. Has to be the same as the one
            used to build the data.

        :type try_loading: bool
        :param try_loading: If set, will attempt to load the whole dataset
            into memory on initialization, leading to faster retrieval times.

        :type save_dataset: str
        :param save_dataset: If the dataset is successfully pre-loaded, attempts
            to pickle the numpy ndarrray into this file.

        :type load_dataset: str
        :param load_dataset: If given, will attempt to load a cPickled numpy
            ndarray from the given file.

        """
        self.icorp = ImagenetCorpus.load(ic_corpus_filename)

        if not dim:
            dim = self.icorp.dim
            logging.debug('Setting image dataset dimension to %d' % dim)

        super(UnsupervisedImagenetCorpusDataset, self).__init__(mm_corpus_filename,
                 dim, test_p, devel_p, serializer, try_loading, save_dataset,
                 load_dataset, profile_loading)
