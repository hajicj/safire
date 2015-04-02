# !/usr/bin/env python
"""
``transformers.py`` is a library that contains various non-NNET SAFIRE
transformer components, mainly used for miscellaneous preprocessing (scaling,
normalization, sigmoid nonlinearity, etc.).

"""
import collections
import itertools
import logging
import operator
import copy
import gensim
from gensim.interfaces import TransformedCorpus
from gensim.similarities import Similarity
import numpy
import scipy.sparse
import safire.datasets.dataset
from safire.utils import gensim2ndarray, IndexedTransformedCorpus
from safire.utils.matutils import sum_gensim_columns
import safire.utils.transcorp

from sklearn.preprocessing import StandardScaler


class NormalizationTransform(gensim.interfaces.TransformationABC):
    """
    Given a corpus, will simply normalize all BOW inputs to sum to
    a normalization constant.
    """
    def __init__(self, C=1.0):
        """Sets the normalization constant."""
        self.C = C

    def __getitem__(self, bow):

        is_corpus, bow = gensim.utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        keys = list(itertools.imap(operator.itemgetter(0), bow))
        values = map(float, list(itertools.imap(operator.itemgetter(1), bow)))

        total = float(sum(values))
        if total == 0.0:
            logging.warn('Item with zero total: %s' % str(bow))
            return bow

        scaling_coefficient = self.C / total
        normalized_values = [float(v) * scaling_coefficient for v in values]
        output = zip(keys, normalized_values)

        return output


class CappedNormalizationTransform(NormalizationTransform):
    """Given a corpus, normalizes each item to a sum such that the largest
    feature value in the corpus is just under 1.0."""
    def __init__(self, corpus, max_target=0.9999):
        """Initializes the scaling."""
        self.max_target = max_target

        # Algorithm
        # What is the correct normalization target?
        # - for each item in corpus, find the constant it could scale to to
        #   reach the max_target
        # - choose the lowest such constant
        coefficients = []
        proposed_constants = []
        for bow in corpus:
            values = map(operator.itemgetter(1), bow)
            max_value = 0.00001
            if len(values) > 0:
                max_value = max(values)
            max_coef = max_target / float(max_value)
            proposed_constant = sum(values) * max_coef

            coefficients.append(max_coef)
            proposed_constants.append(proposed_constant)

        min_safe_constant = min(proposed_constants)
        self.C = min_safe_constant

        logging.info('CappedNormalization: C = %.5f' % self.C)


class MaxUnitScalingTransform(gensim.interfaces.TransformationABC):
    """Scales the vector so that its maximum element is 1."""
    def __getitem__(self, bow):
        keys = list(itertools.imap(operator.itemgetter(0), bow))
        values = list(itertools.imap(operator.itemgetter(1), bow))
        maximum = max(values)
        scaled_values = [float(v) / maximum for v in values]
        output = zip(keys, scaled_values)
        return output


class GlobalUnitScalingTransform(gensim.interfaces.TransformationABC):
    """Scales vectors in the corpus so that the maximum element in the corpus
    is 1. This is to retain proportions between items in an unnormalized
    setting.
    """
    def __init__(self, corpus, cutoff=None):
        """
        :param cutoff: If given, will truncate dataset to this value, prior to
            scaling.
        """
        self.maximum = 0.00001
        for bow in corpus:
            self.maximum = max(self.maximum, max(map(operator.itemgetter(1),
                                                     bow)))
        if cutoff:
            if cutoff < self.maximum:
                self.maximum = float(cutoff)

        logging.info('Found maximum %f with cutoff %f' % (self.maximum, cutoff))

    def __getitem__(self, item, chunksize=None):
        is_corpus, bow = gensim.utils.is_corpus(item)
        if is_corpus:
            return self._apply(item)

        # if isinstance(item, list):
        #     l_output = [ [ (w, max(f, self.maximum) / self.maximum)
        #                    for w, f in i ]
        #                  for i in item ]
        #     return l_output

        output = [ (w, min(f, self.maximum) / self.maximum) for w,f in item ]
        return output

    def _apply(self, corpus, chunksize=1000):

        return TransformedCorpus(self, corpus, chunksize=None)


class SigmoidTransform(gensim.interfaces.TransformationABC):
    """Transforms vectors through a squishing function::

    f(x) = M / (1 + e^(-Kx)) - C

    The defaults are M = 2.0, K = 0.5 and C = 1.0.
    MAKE SURE THAT f(0) = 0 !!!!
    """
    def __init__(self, M=2.0, K=0.5, C=1.0):
        self.M = M
        self.K = K
        self.C = C

    def _fn(self, x):
        return self.M / (1 + numpy.exp(-1.0 * self.K * x)) - self.C

    def __getitem__(self, bow, chunksize=None):
        is_corpus, bow = gensim.utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow, chunksize)

        return [(i, self._fn(x)) for i, x in bow]


class GeneralFunctionTransform(gensim.interfaces.TransformationABC):
    """Transforms each value by the function given at initialization."""
    def __init__(self, fn, multiplicative_coef=1.0, additive_coef=0.0,
                 outer_multiplicative_coef=0.99975, outer_additive_coef=0.0):
        """Will implement the function

        outer_mul_coef * ( fn(mul_coef * X + add_coef) ) + outer_add_coef
        """
        self._fn = fn
        self.mul = multiplicative_coef
        self.add = additive_coef

        self.o_mul = outer_multiplicative_coef
        self.o_add = outer_additive_coef

    def __getitem__(self, bow, chunksize=None):
        is_corpus, bow = gensim.utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow, chunksize)

        if len(bow) == 0:
            logging.debug('Running empty doc through GeneralFunctionTransform.')
        else:
            logging.debug('-- GeneralFunctionTransform. doc length=%d --' % len(bow))

        oK = self.o_mul
        oC = self.o_add
        K = self.mul
        C = self.add
        return [(i, oK * self._fn(K * x + C) + oC) for i, x in bow]


class LeCunnVarianceScalingTransform(gensim.interfaces.TransformationABC):
    """Transforms features so that they all have the same "variance" defined
    by LeCunn, 1998: Efficient BackProp, 4.3, Eq. 13."""
    def __init__(self, corpus, sample=1000, chunksize=1000):
        """Initialized with a corpus. Estimates the scaling coefficients for
        each feature.

        :type corpus: safire.data.sharded_dataset.ShardedDataset

        :param sample: Only use this many first items from the corpus.
            [NOT IMPLEMENTED]

        :param chunksize: Accumulate squared sums by this many.

        """
        self.dim = safire.utils.transcorp.dimension(corpus)

        squared_sums = [ 0.0 for _ in xrange(self.dim) ]

        # This is the only part where we actually have to read the data.
        total_items = 0.0
        for i_group in gensim.utils.grouper(corpus, chunksize=chunksize):
            current_batch = numpy.array([ gensim.matutils.sparse2full(i, self.dim)
                              for i in i_group ])
            total_items += len(current_batch)
            squared_sums += numpy.sum(current_batch ** 2, axis=0)

        self.covariances = squared_sums / total_items
        #self.target_cov = numpy.average(self.covariances)
        self.target_cov = 1.0

        self.coefficients = numpy.sqrt(1/self.covariances) \
                            * numpy.sqrt(self.target_cov)

        logging.info('Average covariance: %f' % self.target_cov)
        logging.info('First few coefficients: %s' % ', '.join(
            map(str, self.coefficients[:10])))

    def __getitem__(self, bow, chunksize=1000):

        is_corpus, bow = gensim.utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow, chunksize=chunksize)

        out = [(i, self.coefficients[i] * x) for i, x in bow]
        #if numpy.random.random() < 0.001:
        #    print 'UCov. Transformation:\n%s\n%s' % (bow[:10], out[:10])
        return out


class TfidfModel(gensim.models.TfidfModel):
    """Overrides _apply to provide IndexedTransformedCorpus, if possible."""
    def _apply(self, corpus, chunksize=None):
        return safire.utils.transcorp.smart_apply_transcorp(self,
                                                            corpus,
                                                            chunksize=chunksize)


class StandardScalingTransformer(gensim.interfaces.TransformationABC):
    """Scales data to zero mean and unit variance."""
    def __init__(self, corpus, with_mean=True, with_variance=True,
                 chunksize=1000):
        """Initializes the transformer.

        :type corpus: gensim.interfaces.CorpusABC
        :param corpus: The corpus that should be scaled.

        :param with_mean: If UNset, will NOT scale to zero mean. (By default:
            do zero-mean scaling.)

        :param with_variance: If UNset, will NOT scale to zero variance.
            (By default: do variance scaling.)
        """
        self.with_mean = with_mean
        self.with_variance = with_variance

        self.sums = numpy.zeros(safire.utils.transcorp.dimension(corpus))
        self.means = numpy.zeros(safire.utils.transcorp.dimension(corpus))
        self.squared_sums = numpy.zeros(safire.utils.transcorp.dimension(corpus))


class Corpus2Dense(gensim.interfaces.TransformationABC):

    def __init__(self, corpus=None, dim=None,
                 dense_throughput=False):
        # Ah, the joys of reliably deriving dimensions and handling missing
        # values.
        if dim is None and corpus is None:
            raise ValueError('Must supply at least one of corpus or dim.')
        proposed_dim = dim
        if corpus is not None:
            try:
                proposed_dim = safire.utils.transcorp.dimension(corpus)
                if dim and proposed_dim != dim:
                    raise ValueError('Derived dimension ({0}) does not '
                                     'correspond to dimension given as argument'
                                     ' ({1}); unsure what to do & quitting.'
                                     ''.format(proposed_dim, dim))
            except ValueError:
                logging.info('Corpus2DenseTransformer: could not derive input '
                             'corpus dimension, using proposed dimension {0}'
                             ''.format(dim))
                if dim is None:
                    raise ValueError('No dimension given and dimension could not'
                                     ' be derived; quitting.')
        self.dim = proposed_dim
        self.dense_throughput = dense_throughput

    def _apply(self, corpus, chunksize=None):
        return safire.utils.transcorp.smart_apply_transcorp(self,
                                                            corpus,
                                                            chunksize=chunksize)

    def __getitem__(self, item):
        """This one should batch-transform lists of gensim vectors on
        slice retrieval, so it's wrong to treat them like corpora -- we can
        apply the transformation on these objects directly. If you want to
        do that, use _apply directly. That's why we don't use the is_corpus
        standard method of identifying gensim corpora."""
        if isinstance(item, gensim.interfaces.CorpusABC):
            return self._apply(item)

        # need to add logic for one item vs. an array of items?
        #print 'Transforming item: {0}'.format(item)
        return gensim2ndarray(item, self.dim)


class SimilarityTransformer(gensim.interfaces.TransformationABC):
    """Adds a pipeline block that will transform the input vector into
    similarities to vectors in a given database.

    This transformer serves as a pipeline-block wrapper for gensim's Similarity
    class (and other similarity index objects someone else may potentially
    code). It views the retrieval process as a transformation: from the space
    of the input vectors and database vectors to the space of similarities to
    the database vectors.

    Usually, we initialize transformers with the same pipeline which we then
    want to run through them. This may be a different case: the similarity index
    will typically be built from some database, but for queries, new vectors
    (albeit from the same vector space as the database vectors) will come as
    queries.
    """
    def __init__(self, index=None, corpus=None, prefix=None, **index_init_args):
        """Initializes the similarity index with the given prefix from the given
        corpus. The ``num_features`` argument is derived from the corpus using
        the usual ``safire.utils.transcorp.dimension()`` function.

        Either initializes directly from a finished Similarity index, or
        constructs the index from a supplied corpus to the given prefix (file).

        Other gensim.similarities.Similarity class init kwargs can be
        provided.
        """
        # Initialize the similarity index
        if index is not None:
            if prefix is not None:
                logging.warn('Initializing SimilarityTransformer with both '
                             'index and prefix. (Index prefix: {0}, supplied '
                             'prefix: {1})'.format(index.output_prefix, prefix))
            if corpus is not None:
                logging.warn('Initializing SimilarityTransformer with both '
                             'index and corpus, corpus will be ignored. '
                             '(Index prefix: {0}, supplied corpus: {1})'
                             ''.format(index.output_prefix,
                                       type(corpus)))
            self.index = index
        else:
            dim = safire.utils.transcorp.dimension(corpus)
            self.index = Similarity(prefix, corpus,
                                    num_features=dim,
                                    **index_init_args)

    def __getitem__(self, item):

        if isinstance(item, gensim.interfaces.CorpusABC) or \
                isinstance(item, safire.datasets.dataset.DatasetABC):
            return self._apply(item)

        return self.index[item]

    def _apply(self, corpus, chunksize=None):

        return safire.utils.transcorp.smart_apply_transcorp(self, corpus)


class ItemAggregationTransform(gensim.interfaces.TransformationABC):
    """The ItemAggregationTransform combines items that map to the same source
    document through the input pipeline's ``id2doc`` mapping.

    The specific method of combining items belonging to the same source document
    is chosen at initialization/subclassed (TBD), by default the aggregator sums
    the individual items by column. Combining a set of items is handled by its
    ``__getitem__()`` method.

    During iteration, the aggregator buffers **consecutive** items that map to
    the same source document. Once the buffer is full, it applies the method of
    aggregation and yields the resulting item.

    .. note::

        This does *not* guarantee that there will be one item per source
        document -- the aggregator combines runs of *consecutive* items from the
        same source doc, so if the input corpus is interleaved, there will be
        multiple items for the same source document. This strategy is chosen to
        make the transformation work without reading the entire input pipeline.

        However, note that this breaks down when CompositeCorpus outputs are
        being aggregated (and indexes are used for flattening)!

    The transformer again has its own corpus, the DocAggregatedCorpus, as it
    alters iteration behavior.

    """
    def __init__(self):
        pass

    def _apply(self, corpus, chunksize=None):
        return ItemAggregationCorpus(self, corpus, chunksize=chunksize)

    def __getitem__(self, itembuffer):
        """In the default implementation, sums the supplied item buffer
        column-wise."""
        if isinstance(itembuffer, gensim.interfaces.CorpusABC):
            return self._apply(itembuffer)

        if isinstance(itembuffer, numpy.ndarray):
            return numpy.sum(itembuffer, axis=0)
        elif isinstance(itembuffer, scipy.sparse.csr_matrix):
            logging.critical('Support for aggregating scipy sparse matrices not'
                             ' implemented yet.')
            raise NotImplementedError()
        else:
            # List of gensim sparse vectors?
            if safire.utils.is_gensim_batch(itembuffer):
            # List of lists of gensim sparse vectors?
                return sum_gensim_columns(itembuffer)
            elif safire.utils.is_list_of_gensim_batches(itembuffer):
                return sum_gensim_columns(list(itertools.chain(*itembuffer)))


class ItemAggregationCorpus(gensim.interfaces.TransformedCorpus):
    """This TransformedCorpus block aggregates items that come from the same
    source document. See :class:`ItemAggregationTransform` for a descriptiont
    of how the corpus behaves.

    This is a utility class that should not be initialized outside the _apply()
    method of ItemAggregationTransform."""
    def __init__(self, obj, corpus, chunksize=None):
        # Chunksize is ignored.
        self.obj = obj
        self.corpus = corpus
        self.dim = safire.utils.transcorp.dimension(corpus)

        self.orig_id2doc = safire.utils.transcorp.get_id2doc_obj(self.corpus)
        self.orig_doc2id = safire.utils.transcorp.get_id2doc_obj(self.corpus)

        # Compute length: it's NOT the number of distinct keys, as the
        # aggregator combines only items from consecutive items with the same
        # source document.
        #
        # In the same cycle, we can pre-compute which original iids will match
        # to which new iid, thus enabling __getitem__ operation.
        self.length = 0
        self.orig2new_iid = collections.defaultdict(int)
        self.new2orig_iid = collections.defaultdict(set)
        prev_doc = None
        for orig_iid, doc in sorted(self.orig_id2doc.items(),
                                    key=operator.itemgetter(0)):
            # If breaking a run of consecutive original IIDs that map to the
            # same source document: add 1 to length, as we will be outputting
            # the combination of buffer items at that point.
            self.orig2new_iid[orig_iid] = self.length
            self.new2orig_iid[self.length].add(orig_iid)
            if doc != prev_doc and prev_doc is not None:
                self.length += 1
            prev_doc = doc
        # Imagine a corpus with items from only one source document.
        # After iterating through it, self.length will never have been
        # incremented. (The new <--> orig iid mappings will use 0 as the new
        # iid for all original items, which is correct.) However, the total
        # length of the aggregated corpus will be 1.
        self.length += 1

        self.doc2id = collections.defaultdict(set)
        self.id2doc = collections.defaultdict(str)

    def __iter__(self):
        itembuffer = []
        output_iid = 0

        # current_docname is the docname we are currently aggregating items for.
        current_docname = None
        for iid, item in enumerate(self.corpus):
            # docname is the document name for the current item.
            docname = self.orig_id2doc[iid]
            if docname == current_docname:
                itembuffer.append(item)
            else:
                if current_docname is None:
                    current_docname = docname
                    itembuffer.append(item)
                else:
                    # These mappings are created for the output item
                    self.doc2id[current_docname].add(output_iid)
                    self.id2doc[output_iid] = current_docname
                    yield self.obj[itembuffer]
                    output_iid += 1
                    current_docname = docname
                    itembuffer = [item]

        # Last set of items
        self.doc2id[current_docname].add(output_iid)
        self.id2doc[output_iid] = current_docname
        yield self.obj[itembuffer]

    def __getitem__(self, item):
        # How to support indexing?
        # - Precompute which original item IDs are
        #   mapped to which aggregated item IDs? Then, if the underlying corpus
        #   is indexable, combine the requested original IDs.
        # print 'Total input iids for source iids {0}: {1}\n{2}' \
        #       ''.format(item,
        #                 len(self.iid2source_iids(item)),
        #                 self.iid2source_iids(item))
        if isinstance(item, int):
            itembuffer = self.iid2items(item)
        elif isinstance(item, slice):
            itembuffer = [self[i] for i in xrange(*item.indices(len(self)))]
            return itembuffer  # Return without aggregation - already aggregated
        elif isinstance(item, list):
            itembuffer = [self[i] for i in item]
            return itembuffer  # Return without aggregateion - already aggregated
        else:
            logging.critical('Cannot aggregate batch over indices of type {0}'
                             ''.format(type(item)))
            raise NotImplementedError()
        logging.debug('Returning itembuffer from request {0}:\n' \
                      'Indices: {1}\nValue: {2}'.format(item,
                                                        self.iid2source_iids(item),
                                                        itembuffer))
        return self.obj[itembuffer]

    def __len__(self):
        return self.length

    def iid2items(self, new_iid):
        """Returns the set of input corpus items that correspond to the given
        new iid.
        """
        orig_iids = sorted(self.new2orig_iid[new_iid])
        if len(orig_iids) == 0:
            return []
        items = self.corpus[orig_iids[0]:orig_iids[-1]]
        return items

    def iid2source_iids(self, new_iid):
        """Returns the list of source corpus iids for the given iid. Can also
        work with a slice or list.
        """
        if isinstance(new_iid, int):
            orig_iids = sorted(self.new2orig_iid[new_iid])
        elif isinstance(new_iid, list):
            orig_iids = list(itertools.chain(*[self.new2orig_iid[i]
                                               for i in new_iid]))
        elif isinstance(new_iid, slice):
            orig_iids = list(itertools.chain(
                *[self.new2orig_iid[i]
                  for i in xrange(*new_iid.indices(len(self)))]
            ))
        else:
            raise TypeError('Can only find new IIDs for an integer, list or '
                            'slice request, not for request of type {0}'
                            ''.format(type(new_iid)))
        return orig_iids


class SplitDocPerFeatureTransform(gensim.interfaces.TransformationABC):
    """Transforms an item into multiple items so that each output item is
    a vector with only one non-zero entry and there is one such vector for
    each non-zero entry in the input vector.

    Note that this does *not* in any way reflect the value at the given
    coordinate, so passing a corpus through this transformer is not equivalent
    to converting for example a document to tokens in case there is a token
    with frequency higher than 1 in the document.
    """
    def __init__(self):
        pass

    def _apply(self, corpus, chunksize=None):
        return SplitDocPerFeatureCorpus(self, corpus, chunksize=chunksize)


class SplitDocPerFeatureCorpus(gensim.interfaces.TransformedCorpus):
    def __init__(self, obj, corpus, chunksize=None):
        if not safire.utils.transcorp.is_fully_indexable(corpus):
            logging.warn('Initializing splitter corpus without a fully'
                         ' indexable input corpus, __getitem__ may not work.'
                         ' (Input corpus type: {0})'.format(type(corpus)))
        self.corpus = corpus
        self.obj = obj
        self.chunksize = chunksize

        # Initialize old to new iid mapping.
        self.new2orig_iids = collections.defaultdict(int)
        self.orig2new_iids = collections.defaultdict(set)

        self.id2doc = collections.defaultdict(str)
        self.doc2id = collections.defaultdict(set)

        # Can we precompute the id2doc/doc2id mapping? No: the iids depend on
        # the size of the input vector.

        self.orig_id2doc = safire.utils.transcorp.get_id2doc_obj(self.corpus)
        self.orig_doc2id = safire.utils.transcorp.get_doc2id_obj(self.corpus)

    def __iter__(self):
        current_iid = 0
        orig_iid = 0
        for item in self.corpus:
            # Gensim corpora
            if isinstance(item, list):
                if safire.utils.is_gensim_vector(item):
                    current_doc = self.orig_id2doc[orig_iid]
                    for entry in item:
                        self.new2orig_iids[current_iid] = orig_iid
                        self.orig2new_iids[orig_iid].add(current_iid)
                        self.id2doc[current_iid] = current_doc
                        self.doc2id[current_doc].add(current_iid)
                        yield [entry]
                        current_iid += 1
                    orig_iid += 1
                # Shouldn't happen..?
                elif safire.utils.is_gensim_batch(item):
                    for row in item:
                        current_doc = self.orig_id2doc[orig_iid]
                        for entry in row:
                            self.new2orig_iids[current_iid] = orig_iid
                            self.orig2new_iids[orig_iid].add(current_iid)
                            self.id2doc[current_iid] = current_doc
                            self.doc2id[current_doc].add(current_iid)
                            yield [entry]
                            current_iid += 1
                        orig_iid += 1
                elif safire.utils.is_list_of_gensim_batches(item):
                    raise TypeError('Got list of gensim batches, expected'
                                    ' at most a gensim batch.')
                else:
                    raise TypeError('Got list, but with unrecognizable content'
                                    ' type. Item: {0}'.format(item))
            # Dense corpora. Note that this transformation is very inefficient
            # for dense corpora.
            elif isinstance(item, numpy.ndarray):
                if len(item.shape) >= 2:
                    for row in item:
                        current_doc = self.orig_id2doc[orig_iid]
                        output = numpy.zeros(row.shape)
                        for idx, entry in enumerate(row):
                            output[idx] = entry
                            self.new2orig_iids[current_iid] = orig_iid
                            self.orig2new_iids[orig_iid].add(current_iid)
                            self.id2doc[current_iid] = current_doc
                            self.doc2id[current_doc].add(current_iid)
                            yield output
                            current_iid += 1
                            orig_iid += 1
                            output[idx] = 0.0
                else:
                    # Yield by coordinate, idividual items
                    output = numpy.zeros(item.shape)
                    current_doc = self.orig_id2doc[orig_iid]
                    for idx, entry in enumerate(row):
                        output[idx] = entry
                        self.new2orig_iids[current_iid] = orig_iid
                        self.orig2new_iids[orig_iid].add(current_iid)
                        self.id2doc[current_iid] = current_doc
                        self.doc2id[current_doc].add(current_iid)
                        yield output
                        current_iid += 1
                        orig_iid += 1
                        output[idx] = 0.0
            else:
                raise TypeError('Unsupported corpus item type: {0}'
                                ''.format(type(item)))
        self.length = current_iid

    def __getitem__(self, key):
        # Problem with __getitem__
        if isinstance(key, slice):
            pass

        if isinstance(key, int):
            pass

    def __len__(self):
        return self.length

