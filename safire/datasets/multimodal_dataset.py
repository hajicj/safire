#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import operator

from gensim.corpora import MmCorpus
import numpy
import theano


#from gensim.corpora.indexedcorpus import IndexedCorpus

#from .dataset import Dataset
from safire.datasets.supervised_dataset import SupervisedDataset
#from vtextcorpus import VTextCorpus
#from .corpus_dataset import UnsupervisedCorpusDataset
from safire.datasets.corpus_dataset import UnsupervisedVTextCorpusDataset
from safire.datasets.corpus_dataset import UnsupervisedImagenetCorpusDataset

logger = logging.getLogger(__name__)


class MultimodalDataset(SupervisedDataset):
    """The MultimodalCorpus is capable of reading both text and image data.
    The image data is assumed to be the output of ImageetCorpus: 4096-dim
    vectors of 32-bit floats. The text data is assumed to be a combined
    VTextCorpus and MmCorpus - see :class:`UnsupervisedVTextCorpusDataset`.

    The dataset will assume that the text and image corpora are aligned - that
    the ``i``-th item in the text corpus corresponds to the ``i``-th item in
    the image corpus, unless a textdoc2imdoc mapping is given. !!!This
    default behavior is soon to be deprecated!!!

    If the textdoc2imdoc mapping is given, the corpus will simulate being
    aligned. **The indexing of the MultimodalDataset is based on the text
    modality:** all images belonging to the same text will be retrieved
    consecutively.

    Because the text2im mapping means that the datasets will not be loaded in
    order, potentially a **lot** of shard-switching may happen on iterating.
    This is severly debillitating to performance. Therefore, two measures are
    taken:

    * On initialization, the text and images are sorted by shard index (texts
      first, images second).
    * When batches are retrieved, they are cached. The cache currently assumes
      the texts fit into memory - there is no limit on cache size. On each mode
      change, the cache is cleared. Only batches that correspond to the current
      *input* mode are cached. (The necessity to cache is currently only felt
      with the multimodal mode, as that is where shard-jumping can happen.)

    """

    def __init__(self, text_mm_filename, text_vt_filename,
                 img_mm_filename, img_ic_filename, dim_text=None, dim_img=None,
                 aligned=False, textdoc2imdoc=None,
                 test_p=0.1, devel_p=0.1, mode=0, shared=False,
                 text_serializer=MmCorpus, img_serializer=MmCorpus):
        """Initializes the dataset - loads corpora, sets devel and test offset.
        Retrieving a batch from the dataset means reading the batch from the
        corpus and converting it to a dense representation.

        Also loads the specified VTextCorpus and uses it to determine the
        output dimension (as the Dictionary size).

        :type text_mm_filename: str
        :param mm_corpus_filename: Where to find the *.mmcorp file previously
            created by ``gensim.corpora.MmCorpus.serialize(fname, corpus)``
            called on a VTextCorpus.

        :type text_vt_filename: str
        :param text_vt_filename: Where to find the *.vtcorp file previously
            created by ``corpus.save(text_vt_filename)`` where ``corpus`` is
            the VTextCorpus serialized to ``text_mm_filename``.

        :type img_mm_filename: str
        :param img_mm_filename: Where to find the *.mmcorp file previously
            created by ``gensim.corpora.MmCorpus.serialize(fname, corpus)``
            called on an ImagenetCorpus.

        :type img_ic_filename: str
        :param img_ic_filename: Where to find the *.iccorp file previously
            created by ``corpus.save(img_ic_filename)`` where ``corpus`` is
            the ImagenetCorpus serialized to ``img_mm_filename``.

        :type dim_text: int
        :param dim: The desired output *y* dimension of the text mode: the size
            of the word vector; number of output columns. This should fit with
            the dimension of the corpus dictionary. (Future plan: implement
            option to truncate to first ``dim`` words.)

        :type dim_img: int
        :param dim_img: Analogous for images: number of image vector output
            dimensions.

        :type aligned: bool
        :param aligned: Should we assume that the corpora are aligned? If set
            to True (default), will disregard the ``textdoc2imdoc`` parameter
            and assume that the text and image corpus are aligned one-to-one:
            the ``i``-th item in one corresponds to the ``i``-th item in the
            other.

        :type textdoc2imdoc: str
        :param textdoc2imdoc: The filename for the map of text document IDs
            to image document IDs. If not supplied, aligned corpora are
            assumed (i.e. 1st text corresponds to 1st image).

        :type test_p: float
        :param test_p: The proportion of documents in the corpus
            used for the test set. The documents will be taken from the end
            of the corpus.

        :type devel_p: float
        :param devel_p: The proportion of documents in the corpus
            used for the development set. The documents will be taken
            from between the training and test sets.

        :type mode: int
        :param mode: Set what kind of dataset the dataset should act like. One
            of ``0``, ``1`` or ``2``.

            * '0' for unsupervised action (returns combined batches of both
              text and image features),

            * ``1`` for text as ``X`` (input), images as ``y`` (response),

            * ``2`` for images as ``X`` (input) and text as ``y`` (response).

            Default setting is ``0``. Note that when mode is ``0``, it is
            assumed that the model is run in a completely unsupervised
            setting, so that the n_out parameter will be supplied to a setup
            function of a model class separately.

        :type shared: False
        :param shared: If set to ``True``, will output batches as Theano shared
            variables. (Default: ``True``)

        :type text_serializer: class froom gensim.interfaces.CorpusABC
        :param text_serializer: The corpus class that should be used to
            de-serialize the text corpus data. Has to be the same as the one
            used to build the data.

        :type img_serializer: class froom gensim.interfaces.CorpusABC
        :param img_serializer: The corpus class that should be used to
            de-serialize the image corpus data. Has to be the same as the one
            used to build the data.

        """
        self.text = UnsupervisedVTextCorpusDataset(text_mm_filename,
                                                   text_vt_filename,
                                                   dim_text, test_p, devel_p,
                                                   serializer=text_serializer)
        self.dim_text = self.text.dim

        self.img = UnsupervisedImagenetCorpusDataset(img_mm_filename,
                                                     img_ic_filename,
                                                     dim_img, test_p, devel_p,
                                                     serializer=img_serializer)
        self.dim_img = self.img.dim

        self.set_mode(mode)
        self.aligned = aligned

        if self.aligned:

            if len(self.text) != len(self.img):
                raise ValueError('Text and image corpora are not of the same '+
                                 'length (text: %d, images: %d)'
                                 % (len(self.text), len(self.img)))

            self._devel_doc_offset = self.text._devel_doc_offset
            if self._devel_doc_offset != self.img._devel_doc_offset:
                raise ValueError('Text and image _devel_doc_offsets do not'+
                             ' match! Are the corpora of the same length? '
                             '(text: %d, image: %d).' % (self.text._devel_doc_offset,
                                               self.img._devel_doc_offset))

            self._test_doc_offset = self.text._devel_doc_offset
            if self._test_doc_offset != self.img._test_doc_offset:
                raise ValueError('Text and image _devel_doc_offsets do not'+
                             ' match! Are the corpora of the same length? '
                             '(text: %d, image: %d).' % (self.text._test_doc_offset,
                                               self.img._test_doc_offset))

            self._text2im_list = zip(range(len(self.text)),
                                      range(len(self.img)))

        else:
            text2im = self._parse_textdoc2imdoc_map(textdoc2imdoc)
            text2im_list = list(self._generate_text2im_list(text2im))
            #opt_t2i_list = self._optimalize_ordering_by_shardidx(text2im_list)
            self._text2im_list = text2im_list

            # set devel and test offsets
            self._test_doc_offset = int(numpy.floor(len(self)
                                                    - (len(self) * test_p)))
            self._devel_doc_offset = int(numpy.floor(self._test_doc_offset
                                                     - (len(self)
                                                               * devel_p)))

            self.text2im_map = {}
            for t, i in self._text2im_list:
                if t in self.text2im_map:
                    self.text2im_map[t].append(i)
                else:
                    self.text2im_map[t] = [i]

        # Should we return shared variables?
        self.shared = shared

        self.cache = {} # Caches some requested batches. Assumes the dataset
                        # will be iterated over, so when the cache is full,
                        # simply doesn't throw away anything.
                        # Cache keys are tuples (b_index, b_size).
        self.cache_size = 0
        self.cache_max_nbytes = 5000000000 # Maximum cache size - 5 * 10^9 B,
                                           # should be set better according to
                                           # some sys.max_mem_param or whatever.

    def n_train_batches(self, batch_size):
        """Computes the number of training batches of the given batch size."""
        return int(numpy.floor(self._devel_doc_offset / batch_size))

    def n_devel_batches(self, batch_size):
        """Computes the number of devel batches of the given batch size."""
        return int(numpy.floor((self._test_doc_offset - self._devel_doc_offset)
                           / batch_size))

    def n_test_batches(self, batch_size):
        """Computes the number of devel batches of the given batch size."""
        return int(numpy.floor((len(self) - self._test_doc_offset)
                           / batch_size))

    def set_mode(self, mode):
        """Controls dataset behavior - what kind of dataset the dataset should
        act like on batch requests. Sets n_in and n_out accordingly.

        :type mode: int
        :param mode: Set what kind of dataset the dataset should act like. One
            of ``0``, ``1`` or ``2``.

            * '0' for unsupervised action (returns combined batches of both
              text and image features),

            * ``1`` for text as ``X`` (input), images as ``y`` (response),

            * ``2`` for images as ``X`` (input) and text as ``y`` (response).

        """
        if mode not in [0,1,2]:
            raise ValueError('Invalid mode set: %s (use one of 0, 1, 2)' % mode)

        if mode == 0:
            self.n_in = self.text.dim + self.img.dim
            self.n_out = None    # This reflects the situation that the model
                                 # is fully unsupervised - the output dimension
                                 # is left to the model initialization.
                                 # The unsupervised model setup class method
                                 # takes care of it.
            self.cache_text = False
            self.cache_img = False
            self.cache_multimodal = True

        elif mode == 1:
            self.n_in = self.text.dim
            self.n_out = self.img.dim

            self.cache_text = True
            self.cache_img = False
            self.cache_multimodal = False

        elif mode == 2:
            self.n_in = self.img.dim
            self.n_out = self.text.dim

            self.cache_text = False
            self.cache_img = True
            self.cache_multimodal = False

        # Invalidates cache.
        self.cache = {}

        self.mode = mode

    def textno2imno(self, textno):
        """Finds the appropriate image numbers for the given text number.
        Returns a list."""
        tid = self.text.vtcorp.id2doc[textno]
        try:
            iids = self.text2im_map[tid]
            imnos = [ self.img.icorp.doc2id[iid] for iid in iids ]
        except KeyError:
            logging.debug('Missed key in textno2imno.')
            logging.debug('Some available keys in t2i map:\n%s' % str(
                self.text2im_map.keys()[:10]))
            logging.debug('Total available keys in t2i map: %d' % len(
                self.text2im_map))
            raise

        return imnos

    def _build_text_batch(self, lbound, batch_size, dtype=theano.config.floatX):
        """Given the first index of a batch and batch size, builds the batch
        from the text corpus.
        """
        if self.cache_text and (lbound, batch_size) in self.cache:
            logging.debug('Retrieving from text cache: (%d, %d)' % (
                lbound,
                batch_size))
            return self.cache[(lbound, batch_size)]

        result = numpy.empty((batch_size, self.text.dim), dtype=dtype)

        logging.debug('Building text batch with lbound %i, size %i.' % (
            lbound,
            batch_size))
        
        for idx, docno in enumerate(xrange(lbound, lbound + batch_size)):
            text_doc, _ = self._text2im_list[docno]
            text_idx = self.text.vtcorp.doc2id[text_doc]

            #doc = self.text[text_idx]
            # Rewritten corpus_dataset classes to dense __getitem__ retval
            result[idx, :] = self.text[text_idx]

        logging.debug('Built text batch: %s' % result)

        if self.cache_text and not self.cache_full() and \
                not (lbound, batch_size) in self.cache:
            logging.debug('Adding to text cache: (%d, %d)' % (lbound, batch_size))
            self.cache_size += result.shape[0] * result.shape[1]
            self.cache[(lbound, batch_size)] = result

        return result

    def _build_image_batch(self, lbound, batch_size, dtype=theano.config.floatX):
        """Given the first index of a batch and batch size, builds the batch
        from the image corpus.
        """
        if self.cache_img and (lbound, batch_size) in self.cache:
            logging.debug('Retrieving from img. cache: (%d, %d)' % (lbound, batch_size))
            return self.cache[(lbound, batch_size)]

        result = numpy.empty((batch_size, self.img.dim), dtype=dtype)

        for idx, docno in enumerate(xrange(lbound, lbound + batch_size)):
            _, img_doc = self._text2im_list[docno]
            img_idx = self.img.icorp.doc2id[img_doc]

            #doc = self.img[img_idx]
            # Rewrote corpus_dataset.py classes do dense __getitem__ retval.
            result[idx, :] = self.img[img_idx]

        if self.cache_img and not self.cache_full() and not (lbound, batch_size) in self.cache:
            self.cache_size += result.shape[0] * result.shape[1]
            logging.debug('Adding to img. cache: (%d, %d)' % (lbound, batch_size))
            self.cache[(lbound, batch_size)] = result

        return result

    def _add_to_cache(self, batch_size, lbound, result):
        """Checks whether the given batch can be added to the cache and if yes,
        adds it. Cache keys are ``(lbound, batch_size)`` tuples.

        :type result: numpy.ndarray
        :param result: The batch to cache.
        """
        if self.cache_multimodal and not self.cache_full() and not (
        lbound, batch_size) in self.cache:
            self.cache_size += result.shape[0] * result.shape[1]
            logging.debug(
                'Adding to mm. cache: (%d, %d)' % (lbound, batch_size))
            self.cache[(lbound, batch_size)] = result

    def _build_multimodal_batch(self, lbound, batch_size,
                                dtype=theano.config.floatX,
                                text_first=True):
        """Given the first index of a batch and batch size, builds the batch
        combined from both the text and image corpora.

        :type text_first: bool
        :param text_first: If set to True, the text dimensions will come before
            the image dimensions in the ordering of the output columns. If set
            to False, the image dimensions will come first. This needs to be
            kept consistent throughout the existence of the dataset! Otherwise,
            results will be garbage.
        """
        if self.cache_multimodal and (lbound, batch_size) in self.cache:
            logging.debug('Retrieving from mm. cache: (%d, %d)' % (lbound, batch_size))
            return self.cache[(lbound, batch_size)]

        result = numpy.empty((batch_size, self.text.dim + self.img.dim),
                             dtype=dtype)

        text_batch = self._build_text_batch(lbound, batch_size, dtype)
        img_batch = self._build_image_batch(lbound, batch_size, dtype)

        if text_first:
            result[:, :text_batch.shape[1]] = text_batch
            result[:, text_batch.shape[1]:] = img_batch
        else:
            result[:, :img_batch.shape[1]] = img_batch
            result[:, img_batch.shape[1]:] = text_batch

        # for idx, docno in enumerate(xrange(lbound, lbound + batch_size)):
        #     # Retrieve corresponding documents from the indexed corpus
        #     text_doc, img_doc = self.__text2im_list[docno]
        #
        #     text_idx = self.text.vtcorp.doc2id[text_doc]
        #     text = self.text[text_idx]
        #
        #     img_idx = self.img.icorp.doc2id[img_doc]
        #     img = self.img[img_idx]
        #
        #     doc = None
        #     if text_first:
        #         doc = safutils.concat_sparse(text, img, self.text.dim,
        #                                      self.img.dim)
        #     else:
        #         doc = safutils.concat_sparse(img, text, self.img.dim,
        #                                      self.text.dim)
        #
        #     result[idx, :] = matutils.sparse2full(doc,
        #                                           self.text.dim + self.img.dim)

        result = result.astype(dtype)

        self._add_to_cache(batch_size, lbound, result)

        return result

    def _build_input_batch(self, lbound, batch_size, dtype=theano.config.floatX):
        """Builds a batch of text or images based on the ``mode`` attribute."""
        if self.mode == 1:
            logging.debug('Building text batch as input batch.')
            output_batch = self._build_text_batch(lbound, batch_size, dtype)
            logging.debug('Built text batch as input batch: %s' % output_batch)
            return output_batch
        
        elif self.mode == 2:
            return self._build_image_batch(lbound, batch_size, dtype)
        
        elif self.mode == 0:
            #logging.warn('Should not use _build_input_batch when unsupervised'+
            #             ' mode (0) is set.')
            return self._build_multimodal_batch(lbound, batch_size, dtype)
        else:
            raise ValueError('Invalid mode set (%s)!' % self.mode)

    def _build_response_batch(self, lbound, batch_size,
                              dtype=theano.config.floatX):
        """Builds a batch of text or images based on the ``mode`` attribute."""
        if self.mode == 2:
            return self._build_text_batch(lbound, batch_size, dtype)
        elif self.mode == 1:
            return self._build_image_batch(lbound, batch_size, dtype)
        elif self.mode == 0:
            logging.warn('Should not use _build_response_batch when unsuper'+
                         'vised mode (0) is set.')
            return self._build_multimodal_batch(lbound, batch_size, dtype)
        else:
            raise ValueError('Invalid mode set (%s)!' % self.mode)

    def _get_batch(self, subset, kind, b_index, b_size,
                   dtype=theano.config.floatX):
        """Retrieves a segment of the data, specified by the arguments.
        The available batch kinds are governed by the ``mode`` attribute.

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

        # When running in a model, b_index will be a Theano tensor.
        lbound = b_index * b_size

        logging.debug('Types: lbound %s, b_index %s, b_size %s, doff %s' % (
            type(lbound), type(b_index), type(b_size),
            type(self._devel_doc_offset)
        ))

        logging.debug('Batch dtype: %s' % dtype)
        logging.debug('Batch subset: %s, kind: %s' % (subset, kind))

        # To make the sanity checks run even when b_index is a Theano var,
        # we need to make some type adjustments...

        # The theano.tensor.var name is overriden as a function, so we cannot
        # check it using isinstance(lbound, theano.tensor.var.TensorVariable)

        #theano.config.compute_test_value = 'warn'
        #lbound.tag.test_value(0.0)

        #logging.debug('Lbound value: %s' % str(lbound.get_value()))
        output_batch = None

        if subset == 'train':            
            if lbound > self._devel_doc_offset:
                raise ValueError('Too high batch index and/or batch size; training dataset has only %d documents.' % (self._devel_doc_offset))

            if kind == 'X':
                logging.debug('Outputtting training X-batch.')
                output_batch = self._build_input_batch(lbound, b_size, dtype)
                logging.debug('Built training X-batch: %s' % output_batch)
            elif kind == 'y':
                output_batch = self._build_response_batch(lbound, b_size, dtype)
            else:
                raise ValueError('Wrong batch kind specified: %s' % kind)

        elif subset == 'devel':
            lbound = lbound + self._devel_doc_offset
            if lbound + b_size > self._test_doc_offset:
                    raise ValueError('Too high batch index and/or batch size (%d, %d); devel dataset has only %d documents.' % (b_index, b_size, self._test_doc_offset - self._devel_doc_offset))

            if kind == 'X':
                output_batch = self._build_input_batch(lbound, b_size, dtype)
            elif kind == 'y':
                output_batch = self._build_response_batch(lbound, b_size, dtype)
            else:
                raise ValueError('Wrong batch kind specified: %s' % kind)

        elif subset == 'test':
            lbound = lbound + self._test_doc_offset
            if lbound + b_size > len(self):
                raise ValueError('Too high batch index and/or batch size (%d, %d); testing dataset has only %d documents.' % (b_index, b_size, len(self) - self._test_doc_offset))

            if kind == 'X':
                output_batch = self._build_input_batch(lbound, b_size, dtype)
            elif kind == 'y':
                output_batch = self._build_response_batch(lbound, b_size, dtype)
            else:
                raise ValueError('Wrong batch kind specified: %s' % kind)


        else:
            raise ValueError('Wrong batch subset specified: %s (datasets only supports \'train\', \'devel\', \'test\').' % subset)

        if self.shared:
            output_batch = theano.shared(output_batch)

        return output_batch

    def _shared_batch(self, batch, borrow=True):
        """Given a numpy array, returns it as a Theano shared variable.
        """
        data = numpy.asarray(batch)
        return theano.shared(data, dtype=theano.config.floatX, borrow=borrow)

    def _parse_textdoc2imdoc_map(self, textdoc2imdoc):
        """Given a file with tab-separated docname/imagename pairs, returns
        a dict with docname keys and list of imagenames values.
        """
        if textdoc2imdoc is None:
            return None

        t2i_map = {}

        with open(textdoc2imdoc) as t2i_handle:

            for i, line in enumerate(t2i_handle):
                text, img = line.strip().split('\t')
                if text not in t2i_map:
                    t2i_map[text] = [img]
                else:
                    t2i_map[text].append(img)

        return t2i_map

    def _generate_text2im_list(self, text2im):
        """Builds a list of text name / image name pairs that covers all
        text-image pairs in the corpus. Expects the self.text2im attribute
        to be filled by output of ``_parse_textdoc2imdoc_map()``."""
        for text in text2im:
            for image in text2im[text]:
                yield text, image

    def _optimalize_ordering_by_shardidx(self, text2im):
        """Sorts the text2im list so that it minimizes the number of calls
        to load_shard."""
        indexed_text2im = []
        t_doc2id = self.text.vtcorp.doc2id
        tdata = self.text.data

        i_doc2id = self.img.icorp.doc2id
        idata = self.img.data

        for t, i in text2im:
            t_idx = t_doc2id[t]
            t_shidx = tdata.shard_by_offset(t_idx)
            i_idx = i_doc2id[i]
            i_shidx = idata.shard_by_offset(i_idx)
            indexed_text2im.append((t, i, t_idx, i_idx))

        sorted_indexed_text2im = sorted(indexed_text2im,
                                        key=operator.itemgetter(2, 3))
        sorted_text2im = map(operator.itemgetter(0,1), sorted_indexed_text2im)
        return sorted_text2im

    def cache_full(self):
        """Checks whether the dataset cache is full."""
        if (self.cache_size * 8) > self.cache_max_nbytes:
            return True
        else:
            logging.debug('Dataset cache is full.')
            return False  # Max cache size: self.cache_max_nbytes bytes

    def __len__(self):
        return len(self._text2im_list)
