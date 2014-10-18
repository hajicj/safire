"""
Provides MultimodalDataset functionality with underlying ShardedDatasets for
both text and images.
"""
import logging
import gensim
from gensim.corpora import MmCorpus
import numpy
import theano
from multimodal_dataset import MultimodalDataset
import safire.utils
from corpus_dataset import UnsupervisedCorpusDataset
from safire.utils.transcorp import dimension
from sharded_dataset import ShardedDataset

__author__ = 'Jan Hajic jr.'


class ShardedMultimodalDataset(MultimodalDataset):
    """
    Implements the MultimodalDataset functionality on top of ShardedDataset-like
    objects for both text and image modality.

    Doesn't re-implement anything but initialization from MultimodalDataset.

    Should be always called using a loader.
    """
    def __init__(self, text_output_prefix, text_vt_filename,
                 img_output_prefix, img_ic_filename,
                 dim_text=None, dim_img=None, textdoc2imdoc=None,
                 test_p=0.1, devel_p=0.1, mode=0, shared=False,
                 text_serializer=MmCorpus, img_serializer=MmCorpus,
                 text_mm_filename=None, img_mm_filename=None, shardsize=4096,
                 overwrite=False):

        # Corpus filenames act the same here. Note that output_prefixes are
        # simply filenames for the saved ShardedDataset objects.

        # The init header is exactly the same as for MultimodalDataset - only
        # instead of mm filenames, the output prefixes are used.

        # Serializers are given to enable reading serialized corpora when the
        # output prefixes are not initialized yet.
        # Additionally, the mm_filenames can be given to facilitate reading
        # data when initializing the datasets.

        self.text = UnsupervisedShardedVTextCorpusDataset(text_output_prefix,
                        text_vt_filename, dim_text, test_p, devel_p,
                        text_serializer, text_mm_filename, shardsize, overwrite)

        self.dim_text = self.text.dim

        self.img = UnsupervisedShardedImagenetCorpusDataset(img_output_prefix,
                        img_ic_filename, dim_img, test_p, devel_p, img_serializer,
                        img_mm_filename, shardsize, overwrite)

        self.dim_img = self.img.dim

        self.set_mode(mode)

        text2im = self._parse_textdoc2imdoc_map(textdoc2imdoc)
        text2im_list = self._generate_text2im_list(text2im)
        opt_t2i_list = self._optimalize_ordering_by_shardidx(text2im_list)
        self._text2im_list = opt_t2i_list

        # Set devel and test offsets
        self._test_doc_offset = int(numpy.floor(len(self) - len(self) * test_p))

        self._devel_doc_offset = int(numpy.floor(self._test_doc_offset
                                                 - len(self) * devel_p))
        self.text2im_map = {}
        for t, i in self._text2im_list:
            if t in self.text2im_map:
                self.text2im_map[t].append(i)
            else:
                self.text2im_map[t] = [i]


        self.shared = shared

        # self.cache = {} # Caches some requested batches. Assumes the dataset
        #                 # will be iterated over, so when the cache is full,
        #                 # simply doesn't throw away anything.
        #                 # Cache keys are tuples (b_index, b_size).
        # self.cache_size = 0
        # self.cache_max_nbytes = 5000000000 # Maximum cache size - 5 * 10^9 B,
        #                                    # should be set better according to
        #                                    # some sys.max_mem_param or whatever.

class UnsupervisedShardedCorpusDataset(UnsupervisedCorpusDataset):
    """A version of UnsupervisedCorpusDataset built around a ShardedDataset
    instead of an IndexedCorpus.
    """
    # Has to override:
    # __init__
    # __getitem__
    # _build_batch
    # _load_dataset
    # get_sparse (will need to extra access the corpus/use full2sparse)

    def __init__(self, output_prefix, dim, test_p=0.1, devel_p=0.1,
                 serializer=MmCorpus, mm_corpus_filename=None, shardsize=4096,
                 overwrite=False):

        self.indexed_corpus = None

        # There are two roles of the indexed corpus: one, to provide docs for
        # shard initialization, and two, to provide sparse items.
        # If not supplied: the initialization will either succeed, if the
        # dataset is being loaded anyway and not initialized from corpus,
        # or fail. On sparse item retrieval, will use backup full2sparse
        # from dataset items (and warn about it).
        if mm_corpus_filename is not None:
            self.indexed_corpus = serializer(mm_corpus_filename)

        # Data are used for dense item retrieval.
        self.data = ShardedDataset(output_prefix, self.indexed_corpus, dim=dim,
                                   test_p=test_p, devel_p=devel_p,
                                   shardsize=shardsize, overwrite=overwrite)

        self.dim = dim
        self.n_in = dim
        self.n_out = None # Unsupervised...

        self.n_docs = len(self.data)

        self.test_p = test_p
        self._test_doc_offset = self.n_docs - int(self.n_docs * self.test_p)

        self.devel_p = devel_p
        self._devel_doc_offset = self._test_doc_offset - int(self.n_docs * self.devel_p)

        # None of the loading B.S. that was in UnsupervisedCorpusDataset.

    def _build_batch(self, lbound, batch_size, dtype=theano.config.floatX):
        """Given the first index of a batch and batch size, builds the batch
        from the corpus. This is easy, because our data directly implements
        slicing."""

        # TODO: implement slicing in ShardedDataset __getitem__ !!!
        #return self.data[lbound:lbound+batch_size]

        # Currently slower than it could be, filling the array one by one
        result = numpy.empty((batch_size, self.dim))
        for idx, docno in enumerate(xrange(lbound, lbound+batch_size)):
            doc = self[docno]
            result[idx, :] = doc

        return result

    def _load_dataset(self, dtype=theano.config.floatX):
        """Returns simply a reference to the current ShardedDataset."""
        return self.data

    def get_sparse(self, idx):
        if self.indexed_corpus is not None:
            return self.indexed_corpus[idx]
        else:
            logging.warn('Retrieving sparse items without a corpus initialized'
                         ' may be inefficient.')
            return gensim.matutils.full2sparse(self.data[idx])

    def __getitem__(self, idx):
        """Retrieves the idx-th item from the dataset in dense form."""
        return self.data[idx]


class UnsupervisedShardedVTextCorpusDataset(UnsupervisedShardedCorpusDataset):
    """Adds VTextCorpus capabilities to UnsupervisedShardedCorpusDataset:
    in addition to loading a ShardedDatasest and optionally a serialized
    MmCorpus, it will load a VTextCorpus (or a transformation) used to
    build the MmCorpus-serialized data (and by extension the ShardedDataset).
    The VTextCorpus is then made available as the ``vtcorp`` member
    of the object.
    """
    def __init__(self, output_prefix, vt_corpus_filename,
                 dim=None, test_p=0.1, devel_p=0.1, serializer=MmCorpus,
                 mm_corpus_filename=None, shardsize=4096, overwrite=False):

        self.loaded_corpus = gensim.utils.SaveLoad.load(vt_corpus_filename)
        logging.info('TextCorpusDataset loaded corpus %s' % str(self.loaded_corpus))
        self.vtcorp = safire.utils.transcorp.bottom_corpus(self.loaded_corpus)
        logging.info('TextCorpusDataset loaded vtcorp %s' % str(self.vtcorp))

        if not dim:
            dim = dimension(self.loaded_corpus)
            logging.debug('Setting text dataset dimension to %d' % dim)

        super(UnsupervisedShardedVTextCorpusDataset, self).__init__(output_prefix,
                dim, test_p, devel_p, serializer, mm_corpus_filename, shardsize,
                overwrite)


class UnsupervisedShardedImagenetCorpusDataset(UnsupervisedShardedCorpusDataset):
    """Adds VTextCorpus capabilities to UnsupervisedShardedCorpusDataset:
    in addition to loading a ShardedDatasest and optionally a serialized
    MmCorpus, it will load a VTextCorpus used to
    build the MmCorpus-serialized data (and by extension the ShardedDataset).
    The VTextCorpus is then made available as the ``vtcorp`` member
    of the object.
    """
    #@profile
    def __init__(self, output_prefix, ic_corpus_filename,
                 dim=None, test_p=0.1, devel_p=0.1, serializer=MmCorpus,
                 mm_corpus_filename=None, shardsize=4096, overwrite=False):

        self.loaded_corpus = gensim.utils.SaveLoad.load(ic_corpus_filename)
        logging.info('ImageCorpusDataset loaded corpus %s' % str(self.loaded_corpus))
        self.icorp = safire.utils.transcorp.bottom_corpus(self.loaded_corpus)
        logging.info('ImageCorpusDataset loaded icorp %s' % str(self.icorp))

        if not dim:
            dim = dimension(self.loaded_corpus)
            logging.debug('Setting image dataset dimension to %d' % dim)

        super(UnsupervisedShardedImagenetCorpusDataset, self).__init__(output_prefix,
                dim, test_p, devel_p, serializer, mm_corpus_filename, shardsize,
                overwrite)