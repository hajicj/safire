"""
This module contains classes that ...
"""
import logging
from gensim.interfaces import TransformationABC, TransformedCorpus
from gensim.utils import is_corpus
import numpy
from safire.utils import IndexedTransformedCorpus
import safire.utils.transcorp
import safire.data.sharded_corpus

__author__ = "Jan Hajic jr."


class Serializer(TransformationABC):
    """This transformer implements a serialization step. It is initialized
    with a corpus, a serializer class (most often a ShardedCorpus) and the
    path to which to serialize the input corpus. Upon _apply, will return
    a SwapoutCorpus - a TransformedCorpus that has two corpora instead of
    a corpus and a transformer obj, and __getitem__ calls to this corpus return
    __getitem__ calls to the second corpus (the one built over the serialized
    data) instead of ``self.obj[self.corpus[key]]``.

    The idea of the Serializer is to have a systematic way of serializing data
    at some fixed points during processing, so that steps that are not changed
    are not run repeatedly but we still retain the whole processing pipeline
    for introspection.
    """
    def __init__(self, corpus, serializer_class, fname,
                 **serializer_init_kwargs):
        """Initializes the serializer transformation.

        :param corpus:

        :param serializer_class: The class to use for serialization. Must
            have the ``serialize()`` classmethod.

        :param fname: The ``fname`` parameter passed to the ``serialize()``
            class method of ``serializer_class``. (For different serializers,
            this filename has a different role, but all need it to operate).

        :param serializer_init_kwargs: Other initialization arguments for the
            serializer.

            When using ShardedCorpus as the serialization class,
            you will need to supply the dimension of a data point. This is
            especially problematic with text corpora, where ShardedCorpus
            derives the dimension from their vocabulary: if the corpus is new
            and should be serialized in this raw form, the dimension will be
            incorrectly derived as 0.
        """
        # This serializes the data
        serializer_class.serialize(fname=fname, corpus=corpus,
                                   **serializer_init_kwargs)
        self.fname = fname  # Used for loading the serialized corpus later.
        self.serializer_class = serializer_class
        self.serialized_data = self.serializer_class.load(self.fname)

    def __getitem__(self, item):

        iscorpus, _ = is_corpus(item)

        if iscorpus:
            return self._apply(item)
        else:
            #raise ValueError('Cannot apply serializer to individual documents.')
            # Will this work?
            return self.serialized_data[item]

            # What happens on a call directly with a data item?

    def _apply(self, corpus, chunksize=None):

        return SwapoutCorpus(self.serializer_class.load(self.fname), corpus)


class SwapoutCorpus(IndexedTransformedCorpus):
    """This class implements a transformation where the outputs of the input
    corpus are swapped for outputs of a different corpus. Use case: instantiated
    by Serializer in order to substitute results transformed through a
    multi-stage pipeline for the pre-computed transformed values. In this
    scenario, the input and output corpora are exactly the same.

    The key idea is that this design will allow us to retain and backtrack
    through the applied transformations while making retrieval faster."""
    def __init__(self, swap, corpus):
        if corpus is None:
            logging.warn('Initializing SwapoutCorpus without a ``corpus``. Note'
                         ' that pattern should only be used for the bottom'
                         ' block of a pipeline.')
        elif len(swap) != len(corpus):
            logging.warn('Swapout corpus: original and swap lengths do not '
                         'match! (original corpus {0}: {1}, swap corpus {2} '
                         ':{3}). Maybe reading serialized corpus from existing '
                         'data, but re-initializing underlying source corpora?'
                         ''.format(type(corpus),
                                   len(corpus),
                                   type(swap),
                                   len(swap)))

        self.obj = swap  # The corpus which gets used instead of the input
                         # corpus.

        self.corpus = corpus
        self.chunksize = None
        self.metadata = False

    def __getitem__(self, key):
        # if isinstance(key, slice):
        #     logging.warn('Are you sure the swapped corpus is sliceable?')
        # print 'Calling __getitem__ on SwapoutCorpus with obj of type {0}' \
        #       'and item {1}'.format(type(self.obj), key)
        out = self.obj[key]
        # print '  SwapoutCorpus.__getitem__: operating on type {0} with ' \
        #       'item {1}'.format(type(out), key)
        # if isinstance(out, numpy.ndarray):
        #     print '              output: shape {0}'.format(out.shape)
        return out

    def __iter__(self):
        for doc in self.obj:
            yield doc

    def save(self, *args, **kwargs):
        logging.info('\n\nSaving SwapoutCorpus with pipeline:\n{0}'
                     ''.format(safire.utils.transcorp.log_corpus_stack(self)))
        super(SwapoutCorpus, self).save(*args, **kwargs)

    def __len__(self):
        return len(self.obj)