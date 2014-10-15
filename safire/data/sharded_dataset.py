#!/usr/bin/env python
"""
Implements a dataset class that stores its data in separate files called
"shards". This is a compromise between speed (keeping the whole dataset
in memory) and memory footprint (keeping the data on disk and reading from it
on demand).
"""
import logging
import os
import cPickle
import gensim
import math
from gensim.interfaces import TransformedCorpus
import numpy
import theano
import time
import safire.utils

__author__ = 'Jan Hajic jr.'

from safire.data.unsupervised_dataset import UnsupervisedDataset

class ShardedDataset(UnsupervisedDataset):
    """
    A dataset that stores its data in separate files called
    "shards". This is a compromise between speed (keeping the whole dataset
    in memory) and memory footprint (keeping the data on disk and reading from
    it on demand). All saving/loading is done using the cPickle mechanism.

    .. note::

      The dataset is **read-only**, there is - as opposed to gensim's Similarity
      class, which works similarly - no way of adding documents to the dataset
      for now.

    On initialization, will read from a corpus and build the dataset. This only
    needs to be done once (and it may take quite a long time):

    >>> icorp = data_loader.load_image_corpus()
    >>> sdata = ShardedDataset(output_prefix, icorp)

    The ``output_prefix`` gives the path to the dataset file. The individual
    shareds are saved as ``output_prefix.0``, ``output_prefix.1``, etc.

    On further initialization with the same ``output_prefix`` (more precisely:
    the output prefix leading to the same file), will load the already built
    dataset unless the ``override`` option is given.

    Internally, to retrieve data, the dataset keeps track of which shard is
    currently open and on a __getitem__ request, either returns an item from
    the current shard, or opens a new one. The shard size is constant, except
    for the last shard.

    TODO: Supports slice notation. [NOT IMPLEMENTED]
    """
    #@profile
    def __init__(self, output_prefix, corpus, dim=None, test_p=0.1, devel_p=0.1,
                 shardsize=4096, overwrite=False):
        """Initializes the dataset. If ``output_prefix`` is not found,
        builds the shards."""
        self.output_prefix = output_prefix
        self.shardsize = shardsize

        self.n_docs = 0

        self.offsets = []
        self.n_shards = 0

        self.dim = dim # This number may change during initialization/loading.

        self.current_shard = None # Current shard (numpy ndarray)
        self.current_shard_n = None
        self.current_offset = None

        logging.info('Initializing shard dataset with prefix %s' % output_prefix)
        if (not os.path.isfile(output_prefix)) or overwrite:
            logging.info('Building from corpus...')
            self.init_shards(output_prefix, corpus, shardsize)
            self.save() # Save automatically, to facillitate re-loading
        else:
            logging.info('Cloning existing...')
            self.init_by_clone()

        # Both methods of initialization initialize self.dim
        self.n_in = self.dim
        self.test_p = test_p
        self._test_doc_offset = self.n_docs - int(self.n_docs * self.test_p)

        self.devel_p = devel_p
        self._devel_doc_offset = self._test_doc_offset \
                                 - int(self.n_docs * self.devel_p)

    def init_shards(self, output_prefix, corpus, shardsize=4096,
                    dtype=theano.config.floatX):
        """Initializes the shards from the corpus."""

        if not gensim.utils.is_corpus(corpus):
            raise ValueError('Cannot initialize shards withot a corpus to read'
                             ' from! (Got: %s)' % str(corpus))

        proposed_dim = self._guess_n_features(corpus)
        if proposed_dim != self.dim:
            if self.dim is None:
                logging.info('Deriving dataset dimension from corpus: %d' % proposed_dim)
            else:
                logging.warn('Dataset dimension derived from input corpus diffe'
                             'rs from initialization argument, using corpus.'
                             '(corpus %d, init arg %d)' % (proposed_dim, self.dim))

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

            for i,doc in enumerate(doc_chunk):
                doc = dict(doc)
                current_shard[i][list(doc)] = list(gensim.matutils.itervalues(doc))

            # Handles the updating as well.
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

        self.dim = temp.dim # To be consistent with the loaded data!

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
            cPickle.dump(shard, pickle_handle, protocol=-1)

        if new_shard:
            self.offsets.append(self.offsets[-1] + len(shard))
            self.n_docs += len(shard)
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
            raise ValueError('Too high offset specified (%i), available docs: %i' % (offset, self.n_docs))
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
            print 'Exception occurred during old shard no. %i removal: %s' % (
                old_shard_n, str(e))
            print 'Attempting to at least move new shards in.'
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
            n_features = corpus.dim
        elif hasattr(corpus, 'dictionary'):
            n_features = len(corpus.dictionary)
        elif hasattr(corpus, 'n_out'):
            n_features = corpus.n_out
        elif hasattr(corpus, 'num_terms'):
            n_features = corpus.num_terms
        elif isinstance(corpus, TransformedCorpus):
            return safire.utils.transcorp.dimension(corpus)
        else:
            raise ValueError('Couldn\'t find number of features, refusing to guess.'
                             '(Type of corpus: %s' % type(corpus))

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
            l_result = [self.get_by_offset(i) for i in offset]
            return l_result

        elif isinstance(offset, slice):
            start = offset.start
            stop = offset.stop
            if stop > self.n_docs:
                raise IndexError('Requested slice offset %d out of range (%d docs)' % (stop, self.n_docs))

            # - get range of shards over which to iterate
            first_shard = self.shard_by_offset(start)

            last_shard = self.n_shards - 1
            if not stop == self.n_docs:
                last_shard = self.shard_by_offset(stop) # This fails on one-past
                # slice indexing; that's why there's a code branch here.


            self.load_shard(first_shard)

            # The easy case: both in one shard.
            if (first_shard == last_shard):
                return self.current_shard[start - self.current_offset:
                                          stop - self.current_offset]

            # The hard case: the slice is distributed across multiple shards
            # - initialize numpy.empty()
            s_result = numpy.empty((stop - start, self.dim),
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
            #  - if in ending shard, thesea re from 0 to (stop - current_offset)
            shard_start = start - self.current_offset
            shard_stop = self.offsets[self.current_shard_n + 1] - self.current_offset

            s_result[result_start:result_stop] = self.current_shard[shard_start:shard_stop]

            # First and last get special treatment, these are in between
            for shard_n in xrange(first_shard+1, last_shard):
                self.load_shard(shard_n)

                result_start = result_stop
                result_stop += self.shardsize
                shard_start = 0
                shard_stop = self.shardsize

                s_result[result_start:result_stop] = self.current_shard[shard_start:shard_stop]

            # Last shard
            self.load_shard(last_shard)
            result_start = result_stop
            result_stop += stop - self.current_offset
            shard_start = 0
            shard_stop = stop - self.current_offset

            s_result[result_start:result_stop] = self.current_shard[shard_start:shard_stop]

            return s_result

        else:
            result = self.get_by_offset(offset)
            return result

    # The obligatory Dataset mehtods.
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
        result = self[lbound:lbound+batch_size]

        return result

    def save(self):
        """Saves itself in clean state (after calling reset()) to the
        output_prefix file."""
        self.reset()
        with open(self.output_prefix, 'wb') as pickle_handle:
            cPickle.dump(self, pickle_handle)

    @classmethod
    def load(cls, output_prefix):

        with open(output_prefix, 'rb') as unpickle_handle:
            dataset = cPickle.load(unpickle_handle)

        return dataset

