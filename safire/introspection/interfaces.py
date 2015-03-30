"""
This module contains classes that allow you to inspect what the pipeline is
actually doing.

The Safire introspection model is technically just another block. However, it's
a pretty strange block - it doesn't transform the data in any way, it just
creates a file for each document (or whatever the output-writing portion is told
to do) and passes on the value unmolested.

The files created by the writers are
saved in the dictionary ``iid2introspection_filename``. Keys in this dictionary
are item IDs (typically just a range from 0 to corpus length), values are
filenames.

While this mechanism is very general, we provide a default implementation that
writes a simple HTML file. More capable writers are
"""
import logging
from gensim.interfaces import TransformationABC, CorpusABC, TransformedCorpus
import math
import numpy
from safire.datasets.dataset import DatasetABC
from safire.introspection.writers import HtmlSimpleWriter
from safire.utils import IndexedTransformedCorpus
from safire.utils.transcorp import get_id2doc_obj, get_doc2id_obj, \
    smart_apply_transcorp, is_fully_indexable

__author__ = "Jan Hajic jr."


class IntrospectionTransformer(TransformationABC):
    """This class applies an IntrospectionCorpus. In this simplest
    implementation, the corpus simply writes the path to the document into an
    html file, but you can supply other Writers.
    """
    def __init__(self, corpus, writer=None, **writer_kwargs):
        #self.id2doc = get_id2doc_obj(corpus)
        #self.doc2id = get_doc2id_obj(corpus)

        self.corpus = corpus

        # This is what actually creates the introspectable files.
        if writer is None:
            writer = HtmlSimpleWriter(**writer_kwargs)
        self.writer = writer
        self.iid2introspection_filename = {}

    def __getitem__(self, item):
        """The IntrospectionTransformer needs two items to operate: first, the
        item itself, and second, the item's ID, so that it can look up
        the original document.

        We need a special TransformedCorpus that will pass this along. This is
        stretching the square brackets a *lot* -- a type check needs to be done.

        :param item:

        :return:
        """
        if isinstance(item, CorpusABC) or isinstance(item, DatasetABC):
            return self._apply(item)

        if not isinstance(item, tuple) or not isinstance(item[0], int):
            raise ValueError('Cannot run IntrospectionTransformer on item type'
                             ' {0}. (Item: {1})'.format(type(item), item))

        iid = item[0]
        value = item[1]

        introspection_file = self.writer.run(iid, value, self.corpus)
        self.iid2introspection_filename[iid] = introspection_file

        return value  # Does not change anything

    def _apply(self, corpus, chunksize=None):
        # Doesn't this need a special (Indexed)TransformedCorpus?
        return EnumeratedTransformedCorpus(self, corpus, chunksize=chunksize)


class EnumeratedTransformedCorpus(IndexedTransformedCorpus):
    """EnumeratedTransformedCorpus is a TransformedCorpus specifically tailored
    for the IntrospectionTransformer.

    The normal TransformedCorpus iterates over ``self.corpus`` as though we
    wrote::

    for item in self.corpus:
        yield self.obj[item]

    Because we often need access to the underlying document during
    introspection, we'll need to see the ID of the incoming document as well.
    So, we instead iterate like this::

    for iid, item in enumerate(self.corpus):
        yield self.obj[(iid, item)]

    The ``__getitem__`` call with a tuple is atrocious, but the idea should be
    clear -- the call to ``self.obj.__getitem__`` translates to a
    ``do_something(iid, item)`` in pseudocode. Imagine rewriting to hide the
    tuple::

    for enumeration_item in enumerate(self.corpus):
        yield self.obj[enumeration_item]

    Looks slightly less terrible now, maybe?
    """
    def __init__(self, obj, corpus, chunksize=None):
        if not isinstance(obj, IntrospectionTransformer):
            raise TypeError('Cannot instantiate IntrospectionCorpus with '
                            'anything else than IntrospectionTransformer, '
                            'instead recieved {0}.'.format(type(obj)))
        self.obj = obj
        self.corpus = corpus
        self.chunksize = chunksize

        if not is_fully_indexable(corpus):
            logging.warn('Working without a fully indexable corpus (type: {0}),'
                         ' cannot guarantee __getitem__ functionality.'
                         ''.format(type(corpus)))

    def __iter__(self):
        """Iterates over (iid, item) pairs. Disregards chunksize information.
        """
        for enumeration_item in enumerate(self.corpus):
            yield self.obj[enumeration_item]

    def __getitem__(self, item):
        """Expects an integer, slice or list. Calls self.__obj__ on (index, doc)
        pairs, where the index runs over all the requested items.
        """
        if isinstance(item, list):
            output = [self.obj[(i, self.corpus[i])] for i in item]
            return output

        elif isinstance(item, slice):
            slice_limit = len(self.corpus)
            if slice.stop is not None:
                slice_limit = item.stop  # This is for slicing uninitialized
                                         # corpora that will only get length
                                         # once iterated over.
            output = [self.obj[(i, self.corpus[i])]
                      for i in xrange(*item.indices(slice_limit))]
            return output

        elif isinstance(item, int):
            return self.obj[(item, self.corpus[item])]

        else:
            raise TypeError('Invalid __getitem__ key specified: type {0}, key '
                            '{1}'.format(type(item), item))
