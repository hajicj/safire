# !/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy

import logging
import operator
from gensim.corpora import TextCorpus

from gensim.interfaces import TransformationABC, TransformedCorpus
from gensim import utils

logger = logging.getLogger(__name__)


class FrequencyBasedTransformer(TransformationABC):
    """Provides functionality for retaining only the top K most frequent
    words in a text corpus.
    """

    def __init__(self, corpus, k, discard_top=0, label=None):
        """Initializes the transformation by obtaining frequencies from the
        corpus, finding the top K most frequent features and building
        a dictionary from them.

        To actually reduce the dataset dimensionality, feature indices are
        re-coded from 0 and a new dictionary is created, together with a dict
        and a reverse dict that code which feature is which in the original
        dictionary.

        TODO: example

        :type corpus: CorpusABC
        :param corpus: The source corpus from which to retain only the
            top K most frequent features.

        :type k: int
        :param k: How many highest-frequency features should be retained. (Used
            to generate transformation label.)

        :type discard_top: int
        param discard_top: How many highest-frequency features should be
            discarded. Typically, these will be little-informative words that
            nevertheless make up a large portion of the data.

        :type label: str
        :param label: A name of the transformation. If supplied, the
        """
        self.k = k
        self.discard_top = discard_top
        self.freqdict = self._build_freqdict(corpus)

        if not label:
            label = self.__default_label()

        self.label = label

        sorted_fv_pairs = sorted(self.freqdict.iteritems(),
                                 key=operator.itemgetter(1),
                                 reverse=True)

        if self.k > len(sorted_fv_pairs):
            logger.warn(
                'Requested more features than are available (%d vs. %d), using all.'
                % (self.k, len(sorted_fv_pairs)))
            self.k = len(sorted_fv_pairs)

        self.allowed_features = frozenset(
            [f[0] for f in sorted_fv_pairs[self.discard_top:self.k]])

        # A dictionary coding original and transformed features onto each other.
        self.orig2transformed = { f : i
                            for i, f in enumerate(list(self.allowed_features)) }
        self.transformed2orig = { i : f
                            for i, f in enumerate(list(self.allowed_features)) }

        # Useful for reporting which features were retained. Maps
        # feature number to feature word.
        if hasattr(corpus, 'dictionary'):
            self.allowed_feature_dict = {
                f : corpus.dictionary[f] for f in self.allowed_features
            }

            # This one allows backtracking using the *new* feature numbers
            self.allowed_new_feature_dict = {
                self.orig2transformed[f] : corpus.dictionary[self.orig2transformed[f]]
                                                for f in self.allowed_features
            }

    def _build_freqdict(self, corpus):
        """Builds a dictionary of feature frequencies in the input corpus.

        :type corpus: gensim.corpora.CorpusABC
        :param corpus: The corpus that is being transformed.
        """

        freqdict = {}
        for item in corpus:
            for feature, value in item:
                if feature in freqdict:
                    freqdict[feature] += value
                else:
                    freqdict[feature] = value

        return freqdict

    def report_features(self):
        """Builds a report of the retained features and their
        frequencies: returns a list of triplets (feature no., feature word,
        feature freq. in corpus) sorted descending by frequency in corpus.
        """
        triplets = [ (f, self.allowed_feature_dict[f], self.freqdict[f])
                     for f in self.allowed_features ]
        sorted_triplets = sorted(triplets, key=operator.itemgetter(2),
                                 reverse=True)
        return sorted_triplets

    def __getitem__(self, bow):
        """Apply transformation from one space to another. In our case: from
        the space of all features, project to the space with only the most
        frequent features."""
        iscorp, corp = utils.is_corpus(bow)
        if iscorp is True:
            return self._apply(bow)

        output = [ (self.orig2transformed[v[0]], v[1])
                   for v in bow if v[0] in self.allowed_features]

        if len(output) == 0:
            logging.warn('Empty item!')

        return output

    def _apply(self, corpus, chunksize=None):
        """Apply transformation in :func:`__getitem__` to the entire corpus.
        Does this by returning gensim's :class:`TransformedCorpus` object that
        applies the transformation over the entire corpus. This is essentially
        a generalization of gensim's VocabTransform class with added facilities
        for backward feature mapping.

        :type corpus: gensim.interfaces.CorpusABC
        :param corpus: The corpus to transform.
        """
        if not isinstance(corpus, TextCorpus):
            logging.warn('Frequency-based transformer applied on non-text' +
                         ' corpus; returning TransformedCorpus.')

            transformed_corpus = TransformedCorpus(self, corpus, chunksize)
            return transformed_corpus

        transformed_corpus = TransformedCorpus(self, corpus, chunksize)
        return transformed_corpus

        # Text corpora: do deep copy, filter transform dictionary

        # Potentially expensive if called on a corpus that stores a lot of
        # information.
        transformed_corpus = deepcopy(corpus)

        # Apply dictionary transformations
        if hasattr(transformed_corpus, 'dictionary'):
            print 'Compactifying dictionary...'
            transformed_corpus.dictionary.filter_tokens(good_ids=list(self.allowed_features))
            transformed_corpus.dictionary.compactify()
            print 'After compactification: %i features.' % len(transformed_corpus.dictionary)
            if hasattr(transformed_corpus, 'allow_dict_updates'):
                transformed_corpus.allow_dict_updates = False

        if hasattr(corpus, 'label'):
            if corpus.label:
                transformed_corpus.label = corpus.label + self.label
            else:
                transformed_corpus.label = self.label

        logging.info('Transformed corpus dictonary has %i features.' % len(transformed_corpus.dictionary))

        return transformed_corpus

    def __default_label(self):
        """Generates the default label for the transformation class, based on
        transformation parameters."""
        return '.top' + str(self.k)

