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
from safire.utils import gensim2ndarray, IndexedTransformedCorpus, \
    is_gensim_vector
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

        # if len(bow) == 0:
        #     logging.debug('Running empty doc through GeneralFunctionTransform.')
        # else:
        #     logging.debug('-- GeneralFunctionTransform. doc length=%d --' % len(bow))

        oK = self.o_mul
        oC = self.o_add
        K = self.mul
        C = self.add
        return [(i, oK * self._fn(K * x + C) + oC) for i, x in bow]

    def _apply(self, corpus, chunksize=None):
        return safire.utils.transcorp.smart_apply_transcorp(self, corpus,
                                                            chunksize=chunksize)


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


class RandomProjectionTransformer(gensim.interfaces.TransformationABC):
    """This transformer randomly chooses a subset of features which to retain
    and discards all others."""
    def __init__(self, k, dim_or_corpus):
        self.k = k
        if isinstance(dim_or_corpus, int):
            self.d = dim_or_corpus
        else:
            self.d = safire.utils.transcorp.dimension(dim_or_corpus)

        if self.k > self.d:
            raise ValueError('Cannot choose {0} features from only '
                             '{1}-dimensional data!'.format(self.k, self.d))

        # Choose the random subset
        self.features = sorted(numpy.random.choice(self.k, range(self.d),
                                                   replace=False))

        self.f_old2new = {f: i for i, f in enumerate(self.features)}
        self.f_new2old = {i: f for i, f in enumerate(self.features)}

        self.dim = self.k

    def __getitem__(self, bow):
        is_corpus, bow = gensim.utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow, chunksize=None)

        if isinstance(bow, numpy.ndarray):
            return bow[:, self.features]
        elif isinstance(bow, scipy.sparse.csr_matrix):
            return bow[:, self.features]
        else:
            # Gensim single vector
            if safire.utils.is_gensim_batch(bow):
                return [self[v] for v in bow]
            else:
                return [(self.f_old2new(f), v) for f, v in bow
                        if f in self.f_old2new]

    def _apply(self, corpus, chunksize=None):
        return safire.utils.transcorp.smart_apply_transcorp(self, corpus,
                                                            chunksize=chunksize)


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
        # print 'Transforming item: {0}'.format(item)
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


class W2IMappingTransform(gensim.interfaces.TransformationABC):
    """The W2IMappingTransformer takes a vector of token iids and any values
    and returns a vector of ``(img_iid, count)`` entries. For each token, the
    image iids associated with that token are returned, with a count for each.
    There are two count strategies available: ``'hard'``, which just adds 1
    per hit, and ``'soft'``, which multiplies this number by the value for the
    respective token, thus taking into account token weights.
    """
    def __init__(self, w2i_mapping, dim, aggregation='hard',
                 runtime_id2word=None):
        """Initializes the transformer. The ``w2i_mapping`` is a dict or other
        structure that will produce on ``__getitem__`` call a list of image
        iids.

        The keys of the w2i mapping can be either ``wid``s, or tokens
        themselves. If we use ``wid``s, we need to keep the word2id mapping
        consistent between the corpus we used to create the mapping and the
        corpus from which the items to transform will come. On the other hand,
        if we use token strings, we will need to translate the ``wid``s to
        tokens, which will be slower: both during initialization and during
        runtime. Nevertheless, the additional flexibility means that we will
        stick with token keys.

        To build the w2i mapping, you will need a token corpus, an image corpus
        and the t2i_indexes that map tokens to images. Additionaly, to convert
        token wids to actual token strings, you'll need a conversion from
        ``iid`` in the token corpus to the individual tokens. (This is handled
        by taking the serialized bottom vtcorp for tokens, calling
        __getitem__(iid) and applying the bottom vtcorp's id2word on the output
        wid from the (wid, 1) freq pair. It is also somewhat slow, as it needs
        to iterate over the entire corpus.)

        Alternately, you can use a document vtcorp and extract all tokens from
        each document. Anyway, this class is not concerned with *how* you obtain
        the w2i mapping. You can use a handy default method from
        ``transcorp.py``: :meth:`build_w2i_mapping_tokencorp` or
         :meth:`build_w2i_mapping_doccorp`

        Dimension of the transformer is described by the total number of iamges
        in the image corpus. This has to be supplied externally, as there is no
        guarantee that the images in the mapping are all the images that were
        available and we should keep the transformed space consistent as the
        space of all available images, because we will be using the source image
        corpus to translate the image dimensions in the image similarity space
        to the docnames of the actual images. (This is the equivalent of
        a vocabulary in the similarity corpus: the WIDs are the iids of the
        original corpus, the words are the image docnames.)
        """
        self.w2i_mapping = w2i_mapping
        self.dim = dim

        if aggregation not in ['hard', 'soft']:
            raise ValueError('Invalid aggregation mode requested: {0} (Use '
                             'either \'hard\', or \'soft\').'
                             ''.format(aggregation))
        self.aggregation = aggregation

        if runtime_id2word is None:
            logging.warn('No id2word provided for runtime wid --> token '
                         'transformation, make sure you supply it before '
                         'running the transformation!')
        self.id2word = runtime_id2word

    def __getitem__(self, item):

        is_corpus, item = gensim.utils.is_corpus(item)
        if is_corpus:
            return self._apply(item, chunksize=None)

        if self.id2word is None:
            raise ValueError('Cannot run __getitem__ without supplying a '
                             'runtime id2word object! (Use the set_id2word_obj'
                             'method.)')

        if isinstance(item, numpy.ndarray):
            raise TypeError('W2IMappingTransformer cannot currently deal with'
                            ' dense inputs, only gensim sparse vectors.')

        output_dict = collections.defaultdict(float)
        for wid, f in item:
            word = self.id2word[wid]
            # TODO: normalize word before querying t2i mapping?
            # May not be necessary, if the t2i mapping is built from data using
            # the same tokenization strategy.
            img_iids = self.w2i_mapping[word]
            for iid in img_iids:
                if self.aggregation == 'hard':
                    output_dict[iid] += 1
                elif self.aggregation == 'soft':
                    output_dict[iid] += f
                else:
                    raise ValueError('Invalid aggregation mode: {0} (Use either'
                                     ' \'hard\', or \'soft\').'
                                     ''.format(self.aggregation))
        out = sorted(output_dict.items(), key=operator.itemgetter(0))
        return out

    def _apply(self, corpus, chunksize=None):
        if self.id2word is None:
            logging.info('Applying W2IMappingTransform: attempting to retireve'
                         ' id2word object from input corpus...')
            self.set_id2word(safire.utils.transcorp.get_id2word_obj(corpus))

        return safire.utils.transcorp.smart_apply_transcorp(self, corpus,
                                                            chunksize=chunksize)

    def set_id2word(self, runtime_id2word):
        """Use this function to manually re-set the id2word mapping. Recommended
        before applying to a corpus (just call :func:`get_id2word_obj` on the
        input corpus)."""
        self.id2word = runtime_id2word


class RetainTopTransform(gensim.interfaces.TransformationABC):
    """This transformer retains top K highest values from the given item.
    (This does NOT change the dimensionality of the pipeline.)
    """
    def __init__(self, k):
        """Initializes the transformer.

        :param k: Retain this many highest values.        """
        self.k = k

    def __getitem__(self, item):
        is_corpus, _ = gensim.utils.is_corpus(item)
        if is_corpus:
            return self._apply(item)

        if not safire.utils.is_gensim_vector(item):
            raise TypeError('RetainTopTransform currently cannot deal with'
                            ' {0}, please convert to gensim.'
                            ''.format(type(item)))
        sorted_by_value = sorted(item, key=operator.itemgetter(1), reverse=True)
        output_items = sorted_by_value[:self.k]
        output = sorted(output_items, key=operator.itemgetter(0))
        return output

    def _apply(self, corpus, chunksize=None):
        return safire.utils.transcorp.smart_apply_transcorp(self,
                                                            corpus,
                                                            cunksize=None)


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
    def __init__(self, average=False):
        self.average = average

    def _apply(self, corpus, chunksize=None):
        return ItemAggregationCorpus(self, corpus, chunksize=chunksize)

    def __getitem__(self, itembuffer):
        """In the default implementation, sums the supplied item buffer
        column-wise."""
        if isinstance(itembuffer, gensim.interfaces.CorpusABC):
            return self._apply(itembuffer)

        return self.sum(itembuffer, average=self.average)

    @staticmethod
    def sum(aggregated_items, average=False):
        """Performs a column-wise sum of the given set of items. If ``averaged``
        is set to True, will also divide the result by the number of items in
        the buffer."""
        if isinstance(aggregated_items, numpy.ndarray):
            total = numpy.sum(aggregated_items, axis=0)
            if average:
                return total / aggregated_items.shape[0]
            else:
                return total
        elif isinstance(aggregated_items, scipy.sparse.csr_matrix):
            logging.critical('Support for aggregating scipy sparse matrices not'
                             ' implemented yet.')
            raise NotImplementedError()
        else:
            # List of gensim sparse vectors?
            if safire.utils.is_gensim_batch(aggregated_items):
                # List of lists of gensim sparse vectors?
                total = sum_gensim_columns(aggregated_items)
            elif safire.utils.is_list_of_gensim_batches(aggregated_items):
                # Dealing with list of gensim batches: flatten it by one level
                flattened_items = list(itertools.chain(*aggregated_items))
                total = sum_gensim_columns(flattened_items)
            else:
                raise ValueError('Cannot deal with the following input: {0}'
                                 ''.format(aggregated_items))
            if average:
                length = float(len(aggregated_items))
                total = [(key, value / length) for key, value in total]
            return total


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

        # What the document -- iid mapping looks like *after* aggregation.
        self.doc2id = collections.defaultdict(set)
        self.id2doc = collections.defaultdict(str)

        logging.debug('Aggregator pre-computing doc/id mappings from corpus {0}...'.format(safire.utils.transcorp.log_corpus_stack(corpus)))
        prev_doc = None
        for orig_iid, doc in sorted(self.orig_id2doc.items(),
                                    key=operator.itemgetter(0)):
            # If breaking a run of consecutive original IIDs that map to the
            # same source document: add 1 to length, as we will be outputting
            # the combination of buffer items at that point.
            self.orig2new_iid[orig_iid] = self.length
            self.new2orig_iid[self.length].add(orig_iid)
            if doc != prev_doc and prev_doc is not None:
                # Compute the id2doc, doc2id mapping as well
                self.id2doc[self.length] = prev_doc
                self.doc2id[prev_doc].add(self.length)
                self.length += 1
            prev_doc = doc
        # Imagine a corpus with items from only one source document.
        # After iterating through it, self.length will never have been
        # incremented. (The new <--> orig iid mappings will use 0 as the new
        # iid for all original items, which is correct.) However, the total
        # length of the aggregated corpus will be 1.
        self.id2doc[self.length] = prev_doc
        self.doc2id[prev_doc].add(self.length)
        self.length += 1

        logging.debug('After pre-computing:\n\tLength: {0}\n\tdoc2id: {1}\n\t'
                      'id2doc: {2}'.format(len(self), self.doc2id, self.id2doc))

    def __iter__(self):
        # Reset doc2id/id2doc mapping
        self.doc2id = collections.defaultdict(set)
        self.id2doc = collections.defaultdict(str)

        itembuffer = []
        output_iid = 0
        # orig_id2doc = safire.utils.transcorp.get_id2doc_obj(self.corpus)

        # current_docname is the docname we are currently aggregating items for.
        current_docname = None

        logging.debug('Aggregator: starting iteration.')
        for iid, item in enumerate(self.corpus):
            # docname is the document name for the current item.
            docname = self.orig_id2doc[iid]
            logging.debug('Processing item with docname: {0}'.format(docname))
            # __id2doc = safire.utils.transcorp.get_id2doc_obj(self.corpus)
            # print 'Original id2doc: id {0}/len {1}, corpus id2doc: id {2}/len {3}'.format(id(orig_id2doc), len(orig_id2doc), id(__id2doc), len(__id2doc))
            # If the underlying corpus is only being built, there are no
            # original docnames to speak of and we'll need to look up the
            # docname from the corpus, not from our snapshot into the corpus
            # at initialization time. This is inefficient, because we have to
            # look up the updated id2doc object each time.
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
                    # print 'Document {0}: Yielding itembuffer of length {1}'.format(current_docname, len(itembuffer))
                    yield self.obj[itembuffer]
                    output_iid += 1
                    current_docname = docname
                    itembuffer = [item]

        # Last set of items
        self.doc2id[current_docname].add(output_iid)
        self.id2doc[output_iid] = current_docname
        # print 'Document {0}: Yielding itembuffer of length {1}'.format(current_docname, len(itembuffer))
        logging.debug('Final doc2id of aggregator: {0}'.format(self.doc2id))
        yield self.obj[itembuffer]

    def __getitem__(self, item):
        # How to support indexing?
        # - Precompute which original item IDs are
        #   mapped to which aggregated item IDs. Then, if the underlying corpus
        #   is indexable, combine the requested original IDs.
        # print 'Total input iids for source iids {0}: {1}\n{2}' \
        #       ''.format(item,
        #                 len(self.iid2source_iids(item)),
        #                 self.iid2source_iids(item))
        logging.debug('Aggregator: retrieving item {0}'.format(item))
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
        # logging.debug('Returning itembuffer from request {0}:\n' \
        #               'Indices: {1}\nValue: {2}'.format(item,
        #                                                 self.iid2source_iids(item),
        #                                                 itembuffer))
        return self.obj[itembuffer]

    def __len__(self):
        return len(self.id2doc)

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


class ReorderingTransform(gensim.interfaces.TransformationABC):
    """This transformation allows arbitrary re-ordering of the underlying
    corpus. No transformation is applied to the content of the data. Note
    that the transformer does not do anything with individual items, its role
    is to create the ReorderingCorpus block.

    To apply a re-ordering, you must first supply the mapping of items from
    the underlying corpus to the new, re-ordered one. This mapping is simply
    a list of indices. If the original corpus looked like this::

    [['A'], ['B'], ['C']]

    and was re-ordered through this map::

    [1, 0, 0, 2, 1, 2]

    the transformed ReorderingCorpus corpus will look like this::

    [['B'], ['A'], ['A'], ['C'], ['B'], ['C']]

    A good use case is for flattening data sources that are not aligned
    (for instance, if multiple news articles use the same image but the image
    appears only once in the image dataset)."""
    def __init__(self, mapping):
        """Initializes the reordering transform."""
        self.mapping = mapping

    def __getitem__(self, item):
        if isinstance(item, gensim.interfaces.CorpusABC):
            return self._apply(item)
        # Duck typing is maybe too risky here?
        # is_corpus, _ = gensim.utils.is_corpus(item)
        # if is_corpus:
        #     return self._apply(item)

        # The basic ReorderingTransformer does not apply any transformation,
        # but maybe some subclasses will?
        return item

    def _apply(self, corpus, chunksize=None):
        return ReorderingCorpus(self, corpus, chunksize)


class ReorderingCorpus(IndexedTransformedCorpus):
    """A pipeline block that handles the re-ordering of items.

    Note that it can only be built over a complete underlying corpus (ideally
    post-serialization)."""
    def __init__(self, obj, corpus, chunksize=None):
        if not isinstance(obj, ReorderingTransform):
            raise TypeError('ReorderingCorpus may only be initialized by a '
                            'ReorderingTransform, got obj of type {0} instead.'
                            ''.format(type(obj)))
        if not safire.utils.transcorp.is_fully_indexable(corpus):
            raise ValueError('Supplied corpu {0} is not fully indexable.'
                             ''.format(corpus))

        self.obj = obj
        self.corpus = corpus

        # Precompute doc2id, id2doc mappings
        self.id2doc = collections.defaultdict(str)
        self.doc2id = collections.defaultdict(set)

        mapping = self.obj.mapping
        orig_id2doc = safire.utils.transcorp.get_id2doc_obj(corpus)
        # orig_doc2id = safire.utils.transcorp.get_doc2id_obj(corpus)
        for iid, orig_iid in enumerate(mapping):
            if orig_iid not in orig_id2doc:
                raise ValueError('Invalid reordering supplied for corpus {0}:'
                                 ' requested iid {1} not available.'
                                 ''.format(corpus, orig_iid))
            docname = orig_id2doc[orig_iid]
            self.id2doc[iid] = docname
            self.doc2id[docname].add(iid)

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self[i] for i in item]
        elif isinstance(item, slice):
            return [self[i] for i in xrange(*item.indices(len(self)))]

        if not isinstance(item, int):
            raise TypeError('Unsupported __getitem__ request type {0} '
                            '(requested item: {1})'.format(type(item), item))

        mapped_id = self.obj.mapping[item]
        retrieved = self.corpus[mapped_id]
        # This currently doesn't do anything,
        # but we might have other subclasses of the ReorderingTransform
        # that reorder items might want  to change the output in some
        # way later.
        output = self.obj[retrieved]
        return output

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.obj.mapping)


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

