"""
Implements a dataset class that stores its data in separate files called
"shards". This is a compromise between speed (keeping the whole dataset
in memory) and memory footprint (keeping the data on disk and reading from it
on demand).
"""
import logging
import os
import cPickle
import math
import numpy
import scipy.sparse as sparse
import theano
import time

import gensim
from gensim.corpora import IndexedCorpus
from gensim.interfaces import TransformedCorpus

import safire.utils
from safire.datasets.unsupervised_dataset import UnsupervisedDataset

__author__ = 'Jan Hajic jr.'


class ShardedDataset(IndexedCorpus, UnsupervisedDataset):
    """This class is designed for situations where you need to train a model
    on dense data, with a large number of iterations (when you need sequential
    and fast access to your data). It should be faster than gensim's other
    IndexedCorpus implementations; check the ``benchmark_datasets.py`` script.

    The dataset stores its data in separate files called
    "shards". This is a compromise between speed (keeping the whole dataset
    in memory) and memory footprint (keeping the data on disk and reading from
    it on demand). All saving/loading is done using the cPickle mechanism.

    .. note::

      The dataset is **read-only**, there is - as opposed to gensim's Similarity
      class, which works similarly - no way of adding documents to the dataset
      (for now).

    On initialization, will read from a corpus and build the dataset. This only
    needs to be done once (and it may take quite a long time):

    >>> icorp = data_loader.load_image_corpus()
    >>> sdata = ShardedDataset(output_prefix, icorp)

    The ``output_prefix`` gives the path to the dataset file. The individual
    shards are saved as ``output_prefix.0``, ``output_prefix.1``, etc.

    On further initialization with the same ``output_prefix`` (more precisely:
    the output prefix leading to the same file), will load the already built
    dataset unless the ``overwrite`` option is given.

    Internally, to retrieve data, the dataset keeps track of which shard is
    currently open and on a ``__getitem__`` request, either returns an item from
    the current shard, or opens a new one. The shard size is constant, except
    for the last shard.

    Gensim interface
    ----------------

    The ShardedDataset simultaneously implements a gensim-style corpus
    interface: the :class:`IndexedCorpus` abstract base class for O(1)
    random-access corpora. (It of course overrides everything
    """

    #@profile
    def __init__(self, output_prefix, corpus, dim=None, test_p=0.1, devel_p=0.1,
                 shardsize=4096, overwrite=False, sparse_serialization=False,
                 sparse_retrieval=False, gensim=False):
        """Initializes the dataset. If ``output_prefix`` is not found,
        builds the shards.

        :type output_prefix: str
        :param output_prefix: The absolute path to the file where the dataset
            object should be saved. The individual shards will be saved as
            ``output_prefix.0``, ``output_prefix.1``, etc.

        :type corpus: gensim.interfaces.CorpusABC
        :param corpus: The source corpus from which to build the dataset.

        :type dim: int
        :param dim: Specify beforehand what the dimension of a dataset item
            should be. This is useful when initializing from a corpus that
            doesn't advertise its dimension, or when it does and you want to
            check that the corpus matches the expected dimension.

        :type test_p: float
        :param test_p: The proportion of the dataset which should be used for
            testing.

        :type devel_p: float
        :param devel_p: The proportion of the dataset which should be used as
            heldout (development) data for validation.

        :type shardsize: int
        :param shardsize: How many data points should be in one shard. More
            data per shard means less shard reloading but higher memory usage
            and vice versa.

        :type overwrite: bool
        :param overwrite: If set, will build dataset from given corpus even
            if ``output_prefix`` already exists.

        :type sparse_serialization: bool
        :param sparse_serialization: If set, will save the data in a sparse
            form (as csr matrices). This is to speed up retrieval when you
            know you will be using sparse matrices.

            This property **should not change** during the lifetime of the
            dataset. (If you find out you need to change from a sparse to
            a dense representation, the best practice is to create another
            ShardedDataset object.)

            [NOT IMPLEMENTED]

        :type sparse_retrieval: bool
        :param sparse_retrieval: If set, will retrieve data as sparse vectors
            (numpy csr matrices). If unset, will return ndarrays.

            Note that retrieval speed for this option depends on how the dataset
            was serialized. If ``sparse_serialization`` was set, then setting
            ``sparse_retrieval`` will be faster. However, if the two settings
            do not correspond, the conversion on the fly will slow the dataset
            down.

            [NOT IMPLEMENTED]

        :type gensim: bool
        :param gensim: If set, will additionally convert the output to gensim
            sparse vectors (list of tuples (id, value)) to make it behave like
            any other gensim corpus. This **will** slow the dataset down.

            [NOT IMPLEMENTED]

        """
        self.output_prefix = output_prefix
        self.shardsize = shardsize

        self.n_docs = 0

        self.offsets = []
        self.n_shards = 0

        self.dim = dim  # This number may change during initialization/loading.

        # Sparse vs. dense serialization and retrieval.
        self._pickle_protocol = -1
        self.sparse_serialization = sparse_serialization
        self.sparse_retrieval = sparse_retrieval
        self.gensim = gensim

        # The "state" of the dataset.
        self.current_shard = None    # The current shard itself (numpy ndarray)
        self.current_shard_n = None  # Current shard is the current_shard_n-th
        self.current_offset = None   # The index into the dataset which
                                     # corresponds to index 0 of current shard

        logging.info('Initializing shard dataset with prefix %s' % output_prefix)
        if (not os.path.isfile(output_prefix)) or overwrite:
            logging.info('Building from corpus...')
            self.init_shards(output_prefix, corpus, shardsize)
            self.save()  # Save automatically, to facillitate re-loading
        else:
            logging.info('Cloning existing...')
            self.init_by_clone()

        # Both methods of initialization initialize self.dim
        self.n_in = self.dim
        self.n_out = self.dim

        self.test_p = test_p
        self._test_doc_offset = self.n_docs - int(self.n_docs * self.test_p)

        self.devel_p = devel_p
        self._devel_doc_offset = self._test_doc_offset \
                                 - int(self.n_docs * self.devel_p)

    def init_shards(self, output_prefix, corpus, shardsize=4096,
                    dtype=theano.config.floatX):
        """Initializes the shards from the corpus."""

        if not gensim.utils.is_corpus(corpus):
            raise ValueError('Cannot initialize shards without a corpus to read'
                             ' from! (Got corpus type: %s)' % type(corpus))

        proposed_dim = self._guess_n_features(corpus)
        if proposed_dim != self.dim:
            if self.dim is None:
                logging.info('Deriving dataset dimension from corpus: '
                             '%d' % proposed_dim)
            else:
                logging.warn('Dataset dimension derived from input corpus diffe'
                             'rs from initialization argument, using corpus.'
                             '(corpus %d, init arg %d)' % (proposed_dim,
                                                           self.dim))

        self.dim = proposed_dim
        self.offsets = [0]

        start_time = time.clock()

        logging.info('Running init from corpus.')

        for n, doc_chunk in enumerate(gensim.utils.grouper(corpus,
                                                           chunksize=shardsize)):
            logging.info('Chunk no. %d at %d s' % (n, time.clock() - start_time))

            current_offset = self.offsets[-1]
            current_shard = numpy.zeros((len(doc_chunk), self.dim),
                                        dtype=dtype)
            logging.debug('Current chunk dimension: %d x %d' % (len(doc_chunk), self.dim))

            for i, doc in enumerate(doc_chunk):
                doc = dict(doc)
                current_shard[i][list(doc)] = list(gensim.matutils.itervalues(doc))

            # Handles the updating as well.
            if self.sparse_serialization:
                current_shard = sparse.csr_matrix(current_shard)

            self.save_shard(current_shard)

        end_time = time.clock()
        logging.info('Built %i shards in %d s.' % (self.n_shards,
                                                   end_time - start_time))

    #@profile
    def init_by_clone(self):
        """Initializes by copying over attributes of another ShardedDataset
        instance saved to the output_prefix given at __init__()."""
        temp = self.__class__.load(self.output_prefix)
        self.n_shards = temp.n_shards
        self.n_docs = temp.n_docs
        self.offsets = temp.offsets

        if temp.dim != self.dim:
            if self.dim is None:
                logging.info('Loaded dataset dimension: %d' % temp.dim)
            else:
                logging.warn('Loaded dataset dimension differs from init arg '
                             'dimension, using loaded dim. '
                             '(loaded %d, init %d)' % (temp.dim, self.dim))

        self.dim = temp.dim  # To be consistent with the loaded data!

    def save_shard(self, shard, n=None, filename=None):
        """Pickles the given shard. If n is not given, will consider the shard
        a new one.

        If ``filename`` is given, will use that file name instead of generating
        one."""
        new_shard = False
        if n is None:
            n = self.n_shards # Saving the *next* one by default.
            new_shard = True

        if not filename:
            filename = self._shard_name(n)
        with open(filename, 'wb') as pickle_handle:
            cPickle.dump(shard, pickle_handle, protocol=self._pickle_protocol)

        if new_shard:
            self.offsets.append(self.offsets[-1] + shard.shape[0])
            self.n_docs += shard.shape[0]
            self.n_shards += 1

    def load_shard(self, n):
        """Loads (unpickles) the n-th shard as the "live" part of the dataset
        into the Dataset object."""

        # No-op if the shard is already open.
        if self.current_shard_n == n:
            pass

        filename = self._shard_name(n)
        if not os.path.isfile(filename):
            raise ValueError('Attemting to load nonexistent shard no. %d' % n)
        with open(filename, 'rb') as unpickle_handle:
            shard = cPickle.load(unpickle_handle)

        self.current_shard = shard
        self.current_shard_n = n
        self.current_offset = self.offsets[n]

    def reset(self):
        """Resets to no shard at all. Used for saving."""
        self.current_shard = None
        self.current_shard_n = None
        self.current_offset = None

    def shard_by_offset(self, offset):
        """Determines which shard the given offset belongs to. If the offset
        is greater than the number of available documents, raises a
        ValueError."""
        if offset >= self.n_docs:
            raise ValueError('Too high offset specified (%d), available docs: %d' % (offset, self.n_docs))
        if offset < 0:
            raise ValueError('Negative offset currently not supported.')

        k = -1
        for i, o in enumerate(self.offsets):
            if o > offset: # Condition should fire for every valid offset,
                           # since the last offset is n_docs (one-past-end).
                k = i - 1 # First offset is always 0, so i is at least 1.
                break

        return k

    def in_current(self, offset):
        """Determines whether the given offset falls within the current
        shard."""
        return (self.current_offset <= offset) \
               and (offset < self.offsets[self.current_shard_n+1])

    def in_next(self, offset):
        """Determines whether the given offset falls within the next shard.
        This is a very small speedup: typically, we will be iterating through
        the data forward. Could save considerable time with a very large number
        of smaller shards."""
        if self.current_shard_n == self.n_shards:
            return False # There's no next shard.
        return (self.offsets[self.current_shard_n+1] <= offset) \
               and (offset < self.offsets[self.current_shard_n+2])

    def resize_shards(self, shardsize):
        """Re-process the dataset to new shard size. This may take pretty long.
        Also, note that you need some space on disk for this one (we're
        assuming there is enough disk space for double the size of the dataset
        and that there is enough memory for old + new shardsize).

        :type shardsize: int
        :param shardsize: The new shard size.
        """

        # Determine how many new shards there will be
        n_new_shards = int(math.floor(self.n_docs / float(shardsize)))
        if self.n_docs % shardsize != 0:
            n_new_shards += 1

        new_shard_names = []
        new_offsets = [0]

        for new_shard_idx in xrange(n_new_shards):
            new_start = shardsize * new_shard_idx
            new_stop = new_start + shardsize

            # Last shard?
            if new_stop > self.n_docs:
                assert new_shard_idx == n_new_shards - 1, 'Shard no. %i that ends at %i over last document (%i) is not the last projected shard (%i)???' % (new_shard_idx, new_stop, self.n_docs, n_new_shards) # Sanity check
                new_stop = self.n_docs

            new_shard = self[new_start:new_stop]
            new_shard_name = self._resized_shard_name(new_shard_idx)
            new_shard_names.append(new_shard_name)

            try:
                self.save_shard(new_shard, new_shard_idx, new_shard_name)
            except Exception:
                # Clean up on unsuccessful resize.
                for new_shard_name in new_shard_names:
                    os.remove(new_shard_name)
                raise

            new_offsets.append(new_stop)

        # Move old shard files out, new ones in. Complicated due to possibility
        # of exceptions.
        old_shard_names = [ self._shard_name(n) for n in xrange(self.n_shards)]
        try:
            for old_shard_n, old_shard_name in enumerate(old_shard_names):
                os.remove(old_shard_name)
        except Exception as e:
            logging.error('Exception occurred during old shard no. %i removal: %s' % (
                old_shard_n, str(e)) + ' Attempting to at least move new shards in.')
            # If something happens with cleaning up - try to at least get the
            # new guys in.
        finally:
            try:
                for shard_n, new_shard_name in enumerate(new_shard_names):
                    os.rename(new_shard_name, self._shard_name(shard_n))
            # If something happens when we're in this stage, we're screwed.
            except Exception as e:
                print e
                raise RuntimeError('Resizing completely failed for some reason.'
                                   ' Sorry, dataset is probably ruined...')
            finally:
                # Sets the new shard stats.
                self.n_shards = n_new_shards
                self.offsets = new_offsets
                self.shardsize = shardsize
                self.reset()

    def _shard_name(self, n):
        """Generates the name for the n-th shard."""
        return self.output_prefix + '.' + str(n)

    def _resized_shard_name(self, n):
        """Generates the name for the n-th new shard temporary file when
        resizing dataset. The file will then be re-named to standard shard name.
        """
        return self.output_prefix + '.resize-temp.' + str(n)

    def _guess_n_features(self, corpus):
        """Attempts to guess number of features in corpus."""
        n_features = None
        if hasattr(corpus, 'dim'):
            # print 'Guessing from \'dim\' attribute.'
            n_features = corpus.dim
        elif hasattr(corpus, 'dictionary'):
            # print 'GUessing from dictionary.'
            n_features = len(corpus.dictionary)
        elif hasattr(corpus, 'n_out'):
            # print 'Guessing from \'n_out\' attribute.'
            n_features = corpus.n_out
        elif hasattr(corpus, 'num_terms'):
            # print 'Guessing from \'num_terms\' attribute.'
            n_features = corpus.num_terms
        elif isinstance(corpus, TransformedCorpus):
            # print 'TransformedCorpus - guessing using transcorp.dimension()'
            return safire.utils.transcorp.dimension(corpus)
        else:
            if not self.dim:
                print self.dim
                raise ValueError('Couldn\'t find number of features, '
                                 'refusing to guess.'
                                 '(Type of corpus: %s' % type(corpus))
            else:
                logging.warn('Couldn\'t find number of features, trusting '
                             'supplied dimension ({0})'.format(self.dim))
                n_features = self.dim
        if self.dim and n_features != self.dim:
            logging.warn('Discovered inconsistent dataset dim (%i) and feature count from corpus (%i). Coercing to corpus dim.' % (self.dim, n_features))
            #n_features = self.dim

        return n_features

    def __len__(self):
        return self.n_docs

    #@profile
    def _ensure_shard(self, offset):
        # No shard loaded
        if self.current_shard is None:
            shard_n = self.shard_by_offset(offset)
            self.load_shard(shard_n)
        # Find appropriate shard, if necessary
        elif not self.in_current(offset):
            if self.in_next(offset):
                self.load_shard(self.current_shard_n + 1)
            else:
                shard_n = self.shard_by_offset(offset)
                self.load_shard(shard_n)

    def get_by_offset(self, offset):
        """As opposed to getitem, this one only accepts ints as offsets."""
        self._ensure_shard(offset)
        result = self.current_shard[offset - self.current_offset]
        return result

    #@profile
    def __getitem__(self, offset):
        """Retrieves the given row of the dataset.

        Slice notation support added, list support for ints added."""
        if isinstance(offset, list):

            # Handle all serialization & retrieval options.
            if self.sparse_serialization:
                l_result = sparse.vstack([self.get_by_offset(i)
                                          for i in offset])
                if self.gensim:
                    l_result = self.__getitem_sparse2gensim(l_result)
                elif not self.sparse_retrieval:
                    l_result = numpy.array(l_result.todense())
            else:
                l_result = numpy.array([self.get_by_offset(i) for i in offset])
                if self.gensim:
                    l_result = self.__getitem_dense2gensim(l_result)
                elif self.sparse_retrieval:
                    l_result = sparse.csr_matrix(l_result)

            return l_result

        elif isinstance(offset, slice):
            start = offset.start
            stop = offset.stop
            if stop > self.n_docs:
                raise IndexError('Requested slice offset'
                                 ' %d out of range (%d docs)' % (stop,
                                                                 self.n_docs))

            # - get range of shards over which to iterate
            first_shard = self.shard_by_offset(start)

            last_shard = self.n_shards - 1
            if not stop == self.n_docs:
                last_shard = self.shard_by_offset(stop) # This fails on one-past
                # slice indexing; that's why there's a code branch here.

            self.load_shard(first_shard)

            # The easy case: both in one shard.
            if first_shard == last_shard:
                s_result = self.current_shard[start - self.current_offset:
                                            stop - self.current_offset]
                # Handle different sparsity settings:
                s_result = self.__getitem_format(s_result)

                return s_result

            # The hard case: the slice is distributed across multiple shards
            # - initialize numpy.zeros()
            s_result = numpy.zeros((stop - start, self.dim),
                                   dtype=self.current_shard.dtype)
            if self.sparse_serialization:
                s_result = sparse.csr_matrix((0, self.dim),
                                             dtype=self.current_shard.dtype)

            # - gradually build it up. We will be using three set of start:stop
            #   indexes:
            #    - into the dataset (these are the indexes the caller works with)
            #    - into the current shard
            #    - into the result

            # Indexes into current result rows. These are always smaller than
            # the dataset indexes by ``start`` (as we move over the shards,
            # we're moving by the same number of rows through the result).
            result_start = 0
            result_stop = self.offsets[self.current_shard_n + 1] - start

            # Indexes into current shard. These are trickiest:
            #  - if in starting shard, these are from (start - current_offset)
            #    to self.shardsize
            #  - if in intermediate shard, these are from 0 to self.shardsize
            #  - if in ending shard, these are from 0
            #    to (stop - current_offset)
            shard_start = start - self.current_offset
            shard_stop = self.offsets[self.current_shard_n + 1] - self.current_offset

            #s_result[result_start:result_stop] = self.current_shard[
            #                                         shard_start:shard_stop]
            s_result = self.__add_to_slice(s_result, result_start, result_stop,
                                           shard_start, shard_stop)

            # First and last get special treatment, these are in between
            for shard_n in xrange(first_shard+1, last_shard):
                self.load_shard(shard_n)

                result_start = result_stop
                result_stop += self.shardsize
                shard_start = 0
                shard_stop = self.shardsize

                s_result = self.__add_to_slice(s_result, result_start,
                                               result_stop, shard_start,
                                               shard_stop)

            # Last shard
            self.load_shard(last_shard)
            result_start = result_stop
            result_stop += stop - self.current_offset
            shard_start = 0
            shard_stop = stop - self.current_offset

            s_result = self.__add_to_slice(s_result, result_start, result_stop,
                                           shard_start, shard_stop)

            s_result = self.__getitem_format(s_result)

            return s_result

        else:
            s_result = self.get_by_offset(offset)
            s_result = self.__getitem_format(s_result)

            return s_result

    def __add_to_slice(self, s_result, result_start, result_stop, start, stop):
        """Adds the rows of the current shard from ``start`` to ``stop``
        into rows ``result_start`` to ``result_stop`` of ``s_result``.

        Operation is based on the self.sparse_serialize setting. If the shard
        contents are dense, then s_result is assumed to be an ndarray that
        already supports row indices ``result_start:result_stop``. If the shard
        contents are sparse, assumes that s_result has ``result_start`` rows
        and we should add them up to ``result_stop``.

        Returns the resulting s_result.
        """
        if (result_stop - result_start) != (stop - start):
            raise ValueError('Result start/stop range different than stop/start'
                             'range (%d - %d vs. %d - %d)' % (result_start,
                                                              result_stop,
                                                              start, stop))

        # Dense data: just copy using numpy's slice notation
        if not self.sparse_serialization:
            try:
                s_result[result_start:result_stop] = self.current_shard[start:stop]
            except ValueError:
                cpdata = self.current_shard[start:stop]
                print 'Copied data: type %s, shape %s' % (type(cpdata),
                                                          str(cpdata.shape))
                print 'S_result: type %s, shape %s' % (type(s_result),
                                                       str(s_result.shape))
                print 'Rstart-Rstop: %d:%d -- start:stop: %d:%d' % (result_start,
                                                                    result_stop,
                                                                    start, stop)
                raise
            return s_result

        # A bit more difficult, we're using a different structure to build the
        # result.
        else:
            if s_result.shape != (result_start, self.dim):
                raise ValueError('Assuption about sparse s_result shape '
                                 'invalid: %d expected rows, %d real rows.' % (
                    result_start, s_result.shape[0]))

            tmp_matrix = self.current_shard[start:stop]
            s_result = sparse.vstack([s_result, tmp_matrix])
            return s_result

    def __getitem_format(self, s_result):
        if self.sparse_serialization:
            if self.gensim:
                s_result = self.__getitem_sparse2gensim(s_result)
            elif not self.sparse_retrieval:
                s_result = numpy.array(s_result.todense())
        else:
            if self.gensim:
                s_result = self.__getitem_dense2gensim(s_result)
            elif self.sparse_retrieval:
                s_result = sparse.csr_matrix(s_result)
        return s_result

    def __getitem_sparse2gensim(self, result):
        """Change given sparse result matrix to gensim sparse vectors.

        Uses the insides of the sparse matrix to make this fast.
        """
        output = [[] for _ in xrange(result.shape[0])]
        for row_idx in xrange(result.shape[0]):
            indices = result.indices[result.indptr[row_idx]:result.indptr[row_idx+1]]
            g_row = [(col_idx, result[row_idx, col_idx]) for col_idx in indices]
            output[row_idx] = g_row
        return output

    def __getitem_dense2gensim(self, result):
        """Change given dense result matrix to gensim sparse vectors."""
        if len(result.shape) == 1:
            output = gensim.matutils.full2sparse(result)
        else:
            output = [gensim.matutils.full2sparse(result[i])
                      for i in xrange(result.shape[0])]
        return output

    # The obligatory Dataset methods.
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

    # Overriding the IndexedCorpus and other corpus superclass methods
    def __iter__(self):
        """Yields items one by one from the dataset.

        This method imitates gensim corpus interface."""
        for i in xrange(len(self)):
            yield self[i]

    def save(self, *args, **kwargs):
        """Saves itself (the wrapper) in clean state (after calling reset())
        to the output_prefix file. If you wish to save to a different file,
        use the ``fname`` argument as the first positional arg."""
        # Can we save to a different file than output_prefix? Well, why not?
        if len(args) == 0:
            args = tuple([self.output_prefix])

        attrs_to_ignore = ['current_shard',
                           'current_shard_n',
                           'current_offset']
        if 'ignore' not in kwargs:
            kwargs['ignore'] = frozenset(attrs_to_ignore)
        else:
            kwargs['ignore'] = frozenset([v for v in kwargs['ignore']]
                                         + attrs_to_ignore)
        super(ShardedCorpus, self).save(*args, **kwargs)
        #
        # self.reset()
        # with open(self.output_prefix, 'wb') as pickle_handle:
        #     cPickle.dump(self, pickle_handle)

    @classmethod
    def load(cls, fname, mmap=None):
        """Loads itself in clean state. You can happily ignore the ``mmap``
        parameter, as the saving mechanism for the dataset is different from
        how gensim saves things in utils.SaveLoad."""
        return super(ShardedCorpus, cls).load(fname, mmap)

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, progress_cnt=1000,
                    metadata=False, **kwargs):
        """Implements a serialization interface a la gensim for the
        ShardedDataset. Do not call directly; use the ``serialize`` method
        instead.

        All this thing does is initialize a ShardedDataset from a corpus
        with the ``output_prefix`` argument set to the ``fname`` parameter
        of this method. The initialization of a ShardedDataset takes care of
        serializing the data (in dense form) to shards.

        Note that you might need some ShardedDataset init parameters, most
        likely the dimension (``dim``). Again, pass these as ``kwargs`` to the
        ``serialize`` method.

        Ignore the parameters id2word, progress_cnt and metadata. They
        currently do nothing and are here only to provide a compatible
        method signature with superclass."""
        ShardedCorpus(fname, corpus, **kwargs)

    @classmethod
    def serialize(serializer, fname, corpus, id2word=None,
                  index_fname=None, progress_cnt=None, labels=None,
                  metadata=False, **kwargs):
        """Iterate through the document stream ``corpus``, saving the documents
        as a ShardedDataset to ``fname``.

        Use this method instead of calling ``save_corpus`` directly.
        You may need to supply some kwargs that are used upon dataset creation
        (namely: ``dim``, unless the dataset can infer the dimension from the
        given corpus).

        Ignore the parameters id2word, index_fname, progress_cnt, labels
        and metadata. They currently do nothing and are here only to
        provide a compatible method signature with superclass."""
        serializer.save_corpus(fname, corpus, id2word=id2word,
                               progress_cnt=progress_cnt, metadata=metadata,
                               **kwargs)
