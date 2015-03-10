"""
This module contains classes that allow you to inspect what the pipeline is
actually doing.

The Safire introspection model is technically just another block. However, it's
a pretty strange block - its output dimension is always 1, and it outputs
a filename with the result of the introspection, whatever you want it to be.

While this mechanism is very general, we provide a default implementation that
writes a simple HTML file.
"""
import logging
from gensim.interfaces import TransformationABC, CorpusABC, TransformedCorpus
from safire.datasets.dataset import DatasetABC
from safire.utils import IndexedTransformedCorpus
from safire.utils.transcorp import get_id2doc_obj, get_doc2id_obj, \
    smart_apply_transcorp

__author__ = "Jan Hajic jr."


class IntrospectionTransformer(TransformationABC):
    """This class applies an IntrospectionCorpus. In this simplest
    implementation, the corpus simply writes the path to the document into an
    html file.
    """
    def __init__(self, corpus, **writer_kwargs):
        self.id2doc = get_id2doc_obj(corpus)
        self.doc2id = get_doc2id_obj(corpus)

        self.dim = 1

        self.writer = HtmlSimpleWriter(**writer_kwargs)   # TODO

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

        if not isinstance(item, tuple) or not isinstance(tuple[0], int):
            raise ValueError('Cannot run IntrospectionTransformer on item type'
                             ' {0}. (Item: {1})'.format(type(item), item))

        iid = item[0]
        document = self.id2doc[iid]
        value = item[1]

        introspection_file = self.writer.write(document, value)
        return introspection_file

    def _apply(self, corpus, chunksize=None):
        # Doesn't this need a special (Indexed)TransformedCorpus?
        return EnumeratedTransformedCorpus(self, corpus, chunksize=chunksize)


class EnumeratedTransformedCorpus(TransformedCorpus):
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

    def __iter__(self):
        """Iterates over (iid, item) pairs. Disregards chunksize information.
        """
        for enumeration_item in enumerate(self.corpus):
            yield self.obj[enumeration_item]