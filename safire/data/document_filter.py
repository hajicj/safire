"""
This module contains classes that ...
"""
import collections
import logging
import copy
import gensim.interfaces
from safire.data.filters.basefilter import BaseFilter
import safire.datasets.dataset
from safire.utils import IndexedTransformedCorpus
from safire.utils.transcorp import get_doc2id_obj, get_id2doc_obj


__author__ = "Jan Hajic jr."


class DocumentFilterTransform(gensim.interfaces.TransformationABC):
    """This class enables removing documents from the data. It contains none
    of the filtering logic itself; its duty is to ensure that the document
    removal is correctly handled. (This is done by initializing the right
    TransformedCorpus block on ``_apply()``, namely
    :class:`DocumentFilterCorpus`.)

    Note that this transformer is somewhat of a special beast -- for example,
    it needs to be able to return "nothing" on a __getitem__ call (transformers
    normally return the transformed version of the input document, this one
    should just play dead; it returns an empty document, which is as close to
    that as we can get). Most of the filtering logic is implemented in its
    associated TransformedCorpus subclass, the DocumentFilterCorpus.
    """
    def __init__(self, flt):
        """Initialize the DocumentFilterTransform transformer.

        :param flt: The object or function that actually decides whether to
            filter or not. It has to be a callable **and** pickleable object!
        """

        if not isinstance(flt, BaseFilter):
            logging.warn('Using a filter of non-BaseFilter type: {0}.'
                         ''.format(type(flt)))
            if not callable(flt):
                raise TypeError('Filter is not callable. (Type: {0})'
                                ''.format(type(flt)))

        self.filter = flt

    def __getitem__(self, item):
        """Performs the "transformation" of the input item. The closest to
        a "correct" transformation of a document that does *not* pass the
        associated document filter is outputting ``None``, so ``__getitem__``
        either outputs the original item (if it passes), or ``None`` (if it
        doesn't).

        The logical alternative of returning an empty document is not what this
        transformer does -- the item should cease to exist in the corpus, not
        just be set to empty; an empty item generalizes to an all-0 item in
        a dense representation, which can/will be requested further down the
        line.

        The intended use-case is applying the transformer on a pipeline, not
        filtering individual documents (you can handle that directly using
        the filter you provided to ``__init__()``. Upon ``_apply()``ing, the
        DocumentFilterTransform constructs a :class:`DocumentFilterCorpus`,
        which handles document removal logic on safire pipelines.

        :param item: CorpusABC, DatasetABC or a document.

        :return: DocumentFilterCorpus, or either the original item, or None.
        """

        if isinstance(item, gensim.interfaces.CorpusABC) or \
                isinstance(item, safire.datasets.dataset.DatasetABC):
            return self._apply(item)

        if self.filter(item):
            return item
        else:
            return None

    def _apply(self, corpus, chunksize=None):

        return DocumentFilterCorpus(self, corpus, chunksize=chunksize)


class DocumentFilterCorpus(IndexedTransformedCorpus):
    """Pipeline block for the DocumentFilterTransform transformation.
    Implements the filtering logic: "manicures" the input data and only lets
    out those that come out of the associated DocumentFilterTransform not None.

    There are two functionalities involved in removing a document from the data.
    First, it needs to be made inaccessible: skipped during iteration, remapped
    to another index on __getitem__ calls, etc. Second, it needs to be removed
    from the ``id2doc`` and ``doc2id`` mappings, and all requests for doc/id
    mappings that pass through the DocumentFilterTransform pipeline step have to
    be handled correctly. This includes several different DocumentFilters on the
    same pipeline!

    """
    def __init__(self, obj, corpus, chunksize=None, dense_throughput=False):

        # Because of how specific this corpus is, doesn't allow usage outside of
        # _apply of DocumentFilterTransform (or a situation that mimics the
        #  _apply call).
        if not isinstance(obj, DocumentFilterTransform):
            raise TypeError('Supplied obj is not a DocumentFilterTransform! '
                            'Instead: {0}'.format(type(obj)))
        self.obj = obj
        self.corpus = corpus
        self.chunksize = chunksize

        self.dense_throughput = dense_throughput

        # These objects provide "source document IDs", persistent identifiers
        # of documents from the data source.
        self.persistent_doc2id = copy.deepcopy(get_doc2id_obj(corpus))
        self.persistent_id2doc = copy.deepcopy(get_id2doc_obj(corpus))

        # These objects provide doc <-> id mapping for the current state of the
        # pipeline. When a document is requested by current ID, this mapping
        # is active.
        self.doc2id = collections.defaultdict(set)
        self.id2doc = collections.defaultdict(str)

        # Used to convert __getitem__ requests to old corpus docIDs. Might have
        # to respond to ``in`` requests and to be wrappable by a KeymapDict.
        self.new2old = dict()
        self.old2new = dict()

        self.n_passed = 0

    def reset(self):
        """Resets the corpus to a pre-iteration state."""
        self.persistent_doc2id = copy.deepcopy(get_doc2id_obj(self.corpus))
        self.persistent_id2doc = copy.deepcopy(get_id2doc_obj(self.corpus))
        self.doc2id = collections.defaultdict(set)
        self.id2doc = collections.defaultdict(str)
        self.new2old = dict()
        self.old2new = dict()
        self.n_passed = 0

    def __iter__(self):
        """Iterates over the input corpus, skipping documents that do not pass
        the DocumentFilter. In the process, the new docname <=> docid mapping
        is created, the docname <=> persistent docid mapping is filtered as
        well.

        Note that every time __iter__ is called, the corpus is re-set.
        The intended pattern"""
        self.reset()
        docid_iterator = iter(list(iter(self.persistent_id2doc)))
        if self.chunksize is not None:
            for chunk in gensim.utils.grouper(self.corpus, self.chunksize,
                                              as_numpy=self.dense_throughput):
                for item in chunk:
                    persistent_docid = docid_iterator.next()
                    transformed = self.obj[item]
                    if transformed:
                        # Update new <=> old mapping
                        self.new2old[self.n_passed] = persistent_docid
                        self.old2new[persistent_docid] = self.n_passed

                        docname = self.persistent_id2doc[persistent_docid]

                        # Building the filtered doc <=> id mapping
                        self.doc2id[docname].add(self.n_passed)
                        self.id2doc[self.n_passed] = persistent_docid

                        self.n_passed += 1
                        yield transformed
                    else:
                        self._remove_docid(persistent_docid)
                        continue
        else:
            for counter, doc in enumerate(self.corpus):
                persistent_docid = docid_iterator.next()
                # print 'Filtering document with persistent ID {0}' \
                #       ', docname {1}' \
                #       ''.format(persistent_docid,
                #                 self.persistent_id2doc[persistent_docid])
                transformed = self.obj[doc]
                if transformed:
                    # Update new <=> old mapping
                    self.new2old[self.n_passed] = persistent_docid
                    self.old2new[persistent_docid] = self.n_passed

                    docname = self.persistent_id2doc[persistent_docid]

                    # Building the filtered doc <=> id mapping
                    self.doc2id[docname].add(self.n_passed)
                    self.id2doc[self.n_passed] = persistent_docid

                    self.n_passed += 1
                    yield transformed
                else:
                    # print '    Removing document! {0}'.format(doc)
                    self._remove_docid(persistent_docid)
                    continue

    def __len__(self):
        # print 'len(self.new2old): {0}, self.n_passed: {1}' \
        #       ''.format(len(self.new2old), self.n_passed)
        return len(self.new2old)

    def __getitem__(self, item):
        """Returns the item-th document of the array-based logic. This supports
        slicing, iterating over a ``range``, etc.

        For persistent ID-based retrieval, see function ``id2doc()`` in
        ``transcorp.py``.

        Note that in order for __getitem__ requests to work properly, the
        pipeline must have been iterated over, otherwise the mapping between
        pre- and post-filtering document IDs will be patchy.
        """
        # Slice retrieval: assumes that the corpus can take list-based
        # retrieval.
        # print 'Calling: {0} with filter {1}'.format(self, self.obj.filter)
        # print 'Retrieving item: {0}'.format(item)
        # print 'Available: {0}'.format(self.new2old)
        if isinstance(item, int):
            return self.corpus[self.new2old[item]]
        if isinstance(item, list):
            return self.corpus[[self.new2old[i] for i in item]]
        if isinstance(item, slice):
            return list(self.corpus[i]
                        for i in xrange(*item.indices(len(self))))

    def _remove_docid(self, docid):

        docname = self.persistent_id2doc[docid]
        try:
            self.persistent_doc2id[docname].remove(docid)
        except KeyError:
            print 'persistent_doc2id[{0}] = {1}' \
                  ''.format(docname, self.persistent_doc2id[docname])
            raise
        del self.persistent_id2doc[docid]
