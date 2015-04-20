import collections
import logging
from gensim.interfaces import TransformationABC

from gensim.utils import is_corpus
import itertools
import numpy
from safire.datasets.dataset import CompositeDataset
from safire.data.composite_corpus import CompositeCorpus

from safire.utils import IndexedTransformedCorpus
from safire.utils import flatten_composite_item
import safire.utils.transcorp
from safire.datasets.dataset import DatasetTransformer, DatasetABC

# TODO: refactor/enhance to work with corpora as well as datasets


class FlattenComposite(TransformationABC):
    """This class flattens a composite dataset into a simple dataset. This
    allows on-the-fly integration of various data sources into unstructured
    batches.

    It is strongly recommended to have serialized all the datasets in the
    CompositeDataset you will be flattening. This won't slow you down much,
    as you might have to access these datasets in some weird and repeated
    patterns anyway, so having a fast random-access version of your data might
    save you time in the long run, even though you needed a pass over the
    entire dataset first.

    .. warn::

        All datasets aggregated in the CompositeDataset being flattened must
        allow list-based retrieval through ``__getitem__``. This *is* the case
        with Datasets that draw from ShardedCorpus-serialized data, as the
        ``__getitem__`` call propagates through the dataset transformation stack
        all the way to the object that is actually supplying the data.

    >>> source1 = DatasetABC([[1], [2], [3]], dim=1)
    >>> source2 = DatasetABC([[-1], [-2], [-3]], dim=1)
    >>> composite = CompositeDataset((source1, source2), names=('source1', 'source2'))
    >>> flatten = FlattenComposite(composite)
    >>> flat = flatten[composite]
    >>> flat[1:3]
    array([[ 2, -2],
           [ 3, -3]])

    There are several options of how to flatten a composite dataset:

    * One-by-one: assumes the datasets are aligned. Most restrictive but
      fastest.
    * Pairing: most general -- assumes nothing, gets an iterable of ``dim``-
      shaped indices and outputs one dataset item per index. (For example,
      to flatten a composite dataset consisting of two simple datasets, use
      indices (x, y).)

    To flatten the dataset using pairing data, you need to supply the pairings
    at transformer initialization time.

    >>> indexes = [(1, 2), (2, 1), (2, 2), (0, 2), (0, 1)]
    >>> flatten_indexed = FlattenComposite(composite, indexes)
    >>> flat = flatten_indexed[composite]
    >>> flat[1:3]
    array([[ 3, -2],
           [ -3, 2]])

    Note that when using the indexed approach with data serialized using
    ShardedCorpus, it may significantly slow operations down, as a lot of
    shard-jumping may be involved.

    The ``id2doc`` and ``doc2id`` mappings are also computed at initialization.

    >>> id2doc = get_id2doc_obj(flat)
    >>> id2doc[1]
    'blah'

    Sometimes you wish to "zip" two corpora together, without combining their
    values into one. For this use case (common for instance in introspection),
    use the ``structured=True`` flag.

    TODO: If that is the case, you can serialize
    the flat dataset using the SerializationTransformer, which allows
    serializing AND retaining the whole pipeline.

    The idea is that you
    load the entire pipeline, in order to have at your disposal everything
    that you used to create a dataset, but you have already serialized your
    data at some advanced point during the pipeline and will not be going
    backwards from that point. This should allow sufficient introspection
    (i.e. going back to the original text corpus for a dictionary...)
    while retaining efficiency (preprocessing steps are not repeated).
    The SerializationTransformer "swaps out" the incoming ``corpus`` for
    a corpus given at initialization time.
    """
    def __init__(self, composite, indexes=None, structured=False):
        """Does nothing, only initializes self.composite and self.indexes.
        Everything happens on-the-fly in __getitem__.

        Future: incorporate a ``precompute`` and/or ``preserialize`` flag."""
        self.composite = composite
        if self.composite.aligned and indexes is None:
            indexes = [composite.as_composite_dim(i)
                       for i in xrange(len(composite))]
        self.indexes = indexes
        self.structured = structured

        if self.indexes is not None and not composite.aligned:
            if not safire.utils.transcorp.is_fully_indexable(composite):
                for corp in composite.corpus:
                    if not safire.utils.transcorp.is_fully_indexable(corp):
                        raise TypeError('Not all data sources are indexable,'
                                        ' cannot perform flattening with '
                                        'indexes. Problem in corpus:\n{0}'
                                        ''.format(safire.utils.transcorp.log_corpus_stack(corp)))
                raise TypeError('Supplied CompositeCorpus itself is not fully '
                                'indexable (sources are OK). Corpus stack:\n{0}'
                                ''.format(safire.utils.transcorp.log_corpus_stack(composite)))
            # Build id2doc mapping. Will contain tuples, based on self.indexes.
            self.id2doc = self.build_composite_id2doc(composite, indexes)
            # What about doc2id? A list of all IDs where the document is
            # used, regardless of which other document it is combined with...
            # Doc2id should trace the influence of a source document through
            # the pipeline.
            self.doc2id = self.build_composite_doc2id(composite, indexes)

    @classmethod
    def build_composite_id2doc(cls, composite, indexes):
        """Builds an id2doc mapping that contains for the i-th index tuple the
        tuple of docnames associated with the data item that the flattened
        dataset returns when asked for the i-th item.

        As opposed to the id2doc mapping of a composite dataset, where there is
        no relationship between the documents making its individual components,
        in a flattened corpus, the individual data sources are already linked
        in fixed items -- only, as opposed to single-source pipeline blocks,
        each item is associated with multiple source documents. (Until a
        composite dataset is flattened, unless it is specified as aligned,
        there's no way of knowing which item from each of its sources will be
        used together with which.)

        :param composite: A composite dataset.

        :param indexes: The indexing scheme for the flattened corpus.

        :return: An id2doc dictionary that returns for each ID the tuple of
            docnames associated with each of the composite data sources.
        """
        # logging.debug('Building composite id2doc for composite:\n{0}'
        #              ''.format(safire.utils.transcorp.log_corpus_stack(composite)))
        id2doc = collections.defaultdict(tuple)
        src_id2docs = []
        for c in composite.corpus:
            logging.debug('Getting id2doc object from corpus:\n{0}'
                          ''.format(safire.utils.transcorp.log_corpus_stack(c)))
            i2d = safire.utils.transcorp.get_id2doc_obj(c)
            src_id2docs.append(i2d)
        for iid, idx in enumerate(indexes):
                id2doc[iid] = tuple(src_id2doc[i]
                                    for src_id2doc, i
                                    in itertools.izip(src_id2docs,
                                                      idx))
        return id2doc

    @classmethod
    def build_composite_doc2id(cls, composite, indexes, id2doc=None):
        """Builds the composite doc2id mapping. For a document, this contains
        all the indices of items in the *flattened* corpus where the given
        source document was used.

        Needs the id2doc mapping to work efficiently. If None is given, will
        construct it (but not return it).

        The method assumes that no two source documents have the same name and
        are only distinguished by which source of the composite data they came
        from!

        :param composite:
        :param indexes:
        :return:
        """
        if id2doc is None:
            id2doc = cls.build_composite_id2doc(composite, indexes)

        output_doc2id = collections.defaultdict(list)
        for i in xrange(len(indexes)):
            current_docs = id2doc[i]
            for doc in current_docs:
                output_doc2id[doc].append(i)
        return output_doc2id

    def _apply(self, dataset, chunksize=None):
        return FlattenedDatasetCorpus(self, dataset, structured=self.structured)

    def __getitem__(self, item):

        iscorpus, _ = is_corpus(item)

        if iscorpus or isinstance(item, DatasetABC):
            return self._apply(item)
        else:
            raise ValueError('Cannot apply flatten_composite to individual '
                             'documents.')


class FlattenedDatasetCorpus(IndexedTransformedCorpus):

    def __init__(self, flatten, corpus, structured=False):
        """
        :param flatten:

        :param corpus:

        :param structured: If set, will *not* merge the retrieved items into
            one. Useful for obtaining the indexing pairs as separate entities.
            However, the only currently useful use-case is as the underlying
            data for introspection by a HtmlStructuredFlattenedWriter.

        :return:
        """
        self.obj = flatten
        self.indexes = flatten.indexes
        self.corpus = corpus  # A composite corpus

        self.dim = self.derive_dimension(self.corpus)
        self.n_in = self.dim
        self.n_out = self.dim

        self.structured = structured

        self.chunksize = None
        self.id2doc = flatten.build_composite_id2doc(self.corpus, self.indexes)
        self.doc2id = flatten.build_composite_doc2id(self.corpus, self.indexes,
                                                     self.id2doc)

    def __getitem__(self, item):
        """This is where on-the-fly construction of the flattened dataset
        happens.

        This method is NOT format-agnostic, it assumes its inputs and outputs
        are in numpy ndarray format.

        :param item: An index or slice.

        :return: numpy.ndarray
        """
        # print '...flattening item: {0}'.format(item)
        if self.indexes is None or self.corpus.aligned:
            # print '...flattening from aligned'
            retrieved = self.corpus[item]
            output = self.item2flat(retrieved, nostack=self.structured)
        else:
            if isinstance(item, list):
                return [self[i] for i in item]
            if isinstance(item, slice):
                return [self[i] for i in xrange(*item.indices(len(self)))]

            # Possibly inefficient
            # print 'Item: {0}'.format(item)
            indexes = self.indexes[item]
            try:
                idxs_by_dataset = map(list, zip(*indexes))
            except TypeError:
                # Single item retrieved
                indexes = [indexes]
                idxs_by_dataset = map(list, zip(*indexes))
            # print 'Flattening with indexes {0}'.format(idxs_by_dataset)
            # logging.debug('Indexes: {0}, by dataset: {1}'
            #              ''.format(indexes, idxs_by_dataset))
            retrieved = []
            for dataset, idxs in zip(self.corpus.corpus, idxs_by_dataset):
                # logging.debug(' Retrieving: dataset {0}, '
                #               'idxs {1}'.format(dataset, idxs))

                # Depends here on ability of SwapoutCorpus serialized by
                # ShardedCorpus to deliver numpy ndarrays from lists of indices.
                partial = dataset[idxs]
                # ('Partial for dataset {0}, idxs {1}: {2}'
                #               ''.format(dataset, idxs, partial))
                if isinstance(partial[0], numpy.ndarray):
                    partial = numpy.array(partial)

                # print 'Partial shape: {0}'.format(partial.shape)
                retrieved.append(partial)

            # Here depends on availability of ``shape`` instance attribute
            #logging.debug('Retrieved shapes: {0}'
            #              ''.format([r.shape for r in retrieved]))
            output = self.item2flat(retrieved, nostack=self.structured)

        # Problem: flattening is outputting

        # logging.debug('flattening __getitem__ output: {0}'.format(output))
        # print 'flattening __getitem__ output: {0}'.format(output)
        return output

    def derive_dimension(self, composite):
        return FlattenedDatasetCorpus.flattened_dimension(
            safire.utils.transcorp.dimension(composite))

    def __len__(self):
        if self.indexes is not None:
            return len(self.indexes)
        else:
            return len(self.corpus)

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    @staticmethod
    def item2flat(item, nostack=False):
        """Flattens a (recursive) tuple of numpy ndarrays/scipy sparse matrices
        and stacks them next to each other (i.e.: rows stay rows, columns
        change).

        >>> x = numpy.array([[1], [2], [3], [4]])
        >>> y = numpy.array([[-1], [-3], [-5], [-7]])
        >>> z = numpy.array([[10, 20], [11, 21], [12, 22], [13, 23]])
        >>> item = (x, (y, z))
        >>> FlattenedDatasetCorpus.item2flat(item)

        """
        # print 'item2flat: item {0}, nostack {1}'.format(item, nostack)
        output = list(flatten_composite_item(item))
        if not nostack:
            output = numpy.hstack(output)
        # print 'item2flat output: {0}'.format(output)
        return output

    @staticmethod
    def flattened_dimension(composite_dim):

        total = 0
        logging.debug('Composite dim: {0}'.format(composite_dim))
        for d in composite_dim:
            if isinstance(d, tuple):
                total += FlattenedDatasetCorpus.flattened_dimension(d)
            else:
                total += d
        return total


