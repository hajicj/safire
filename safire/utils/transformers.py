# !/usr/bin/env python
"""
``transformers.py`` is a library that contains various non-NNET SAFIRE
transformer components, mainly used for miscellaneous preprocessing (scaling,
normalization, sigmoid nonlinearity, etc.).

"""
import itertools
import logging
import operator
import gensim
from gensim.interfaces import TransformedCorpus
from gensim.similarities import Similarity
import numpy
#from safire.utils.transcorp import dimension
import safire.datasets.dataset
from safire.utils import gensim2ndarray, IndexedTransformedCorpus
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
    def __init__(self, corpus, prefix):
        # Initialize the similarity index
        dim = safire.utils.transcorp.dimension(corpus)
        self.index = Similarity(prefix, corpus,
                                num_features=dim)

    def __getitem__(self, item):

        if isinstance(item, gensim.interfaces.CorpusABC) or \
                isinstance(item, safire.datasets.dataset.DatasetABC):
            return self._apply(item)

        return self.index[item]

    def _apply(self, corpus, chunksize=None):

        return safire.utils.transcorp.smart_apply_transcorp(self, corpus)