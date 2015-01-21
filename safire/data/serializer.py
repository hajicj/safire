"""
This module contains classes that ...
"""
import logging
from gensim.interfaces import TransformationABC, TransformedCorpus

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

        :param fname:
        :param serializer_init_kwargs:
        :return:
        """
        # This serializes the data
        serializer_class.serialize(fname=fname, corpus=corpus,
                                   **serializer_init_kwargs)
        self.fname = fname  # Used for loading the serialized corpus later.
        self.serializer_class = serializer_class

    def _apply(self, corpus, chunksize=None):

        return SwapoutCorpus(corpus, self.serializer_class.load(self.fname))


class SwapoutCorpus(TransformedCorpus):
    """This class implements a transformation where the outputs of the input
    corpus are swapped for outputs of a different corpus. Use case: instantiated
    by Serializer in order to substitute results transformed through a
    multi-stage pipeline for the pre-computed transformed values. In this
    scenario, the input and output corpora are exactly the same.

    The key idea is that this design will allow us to retain and backtrack
    through the applied transformations while making retrieval faster."""
    def __init__(self, swap, corpus):
        self.obj = swap  # The corpus which gets used instead of the input
                         # corpus.
        self.corpus = corpus
        self.chunksize = None
        self.metadata = False

    def __getitem__(self, key):
        return self.obj[key]

    def __iter__(self):
        for doc in self.obj:
            yield doc