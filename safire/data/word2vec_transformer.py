"""
This module contains classes that ...
"""
import codecs
import cPickle
import logging
import itertools
import numpy
import re
import sys
import time

from memory_profiler import profile

import gensim
from gensim.corpora import TextCorpus
from gensim.interfaces import TransformationABC, TransformedCorpus

from safire.utils import total_size
import safire.utils.transcorp

__author__ = "Jan Hajic jr."

#: Used in stripping lemmatization artifacts from queries.
deUFALize_regex = re.compile('\-[0-9]+$')


class Word2VecTransformer(TransformationABC):
    """This class applies learned embeddings on a text corpus.

    Each document is obtained using a ``max`` operation over each dimension
    of the embeddings over all words in the document. If a word is not found in
    the embedding dictionary, it is left out (equivalent to setting it to
    all 0's).

    Note that this transformer will discard frequency information and also make
    it impossible to trace individual words back to the data.

    To operate, the Word2VecTransformer needs a reverse dictionary that
    transforms word IDs back to the original words, as the embeddings are
    accessed by the word they represent, so as for the Word2VecTransformer
    to be independent on the corpus it is supposed to process. When processing
    a corpus, it uses the :function:`safire.utils.transcorp.id2word` function
    to convert the word ID to a word.

    There are two ways of supplying the dictionary: at initialization time,
    and when calling the :meth:`__getitem__` method on a corpus, if the corpus
    is associated with the dictionary. The ``id2word`` method is used when no
    dictionary is supplied.
    """

    def __init__(self, embeddings, id2word, op='max', lowercase=False,
                 dense=False,  deUFALize=True, from_pickle=True):
        """Initializes the embeddings.

        :param embeddings: A filename or a handle to a word vector file that
            contains the embeddings. The expected format is one word per line,
            separated by whitespace or tabspace, with the first item being
            the word itself and the other items are the embedding vector.
            Additionally, the first line of the file contains two fields:
            the size of the vocabulary and the dimension of the embedding.

        :param id2word: A dictionary that maps word IDs to the
            words themselves. If not supplied, will rely on the processed
            corpus to be processable by the :meth:`utils.transcorp.id2word`
            method. All that is required is that it has a ``__getitem__``
            method.

        :param op: The operation to apply when combining word vectors into
            a document vector. Accepts ``max``, ``sum``, ``average``.

        :param lowercase: If this flag is set, will lowercase ``id2word``
            values that are used as search keys for the embeddings. This is
            basically a standardization step.

        :param dense: If this flag is set, will output dense vectors
            instead of gensim-style sparse vectors.

        :param from_pickle: If this flag is set, will consider the
            ``embeddings`` parameter to point to a pickled embeddings dict
            instead of a vectors file.

        :param deUFALize: If this flag is set, will assume that the id2word
            values are UFAL lemmas that need to be further standardized to
            correspond to basic forms that are in the word2vec embeddings dict.
        """
        if from_pickle:
            if not isinstance(embeddings, str):
                raise TypeError('Embeddings requested from pickle, but not '
                                'supplied as a string.')
            with open(embeddings, 'rb') as ehandle:
                self.embeddings = cPickle.load(ehandle)
                self.n_out = len(self.embeddings[self.embeddings.keys()[0]])
        elif isinstance(embeddings, str):
            with codecs.open(embeddings, 'r', 'utf-8') as embeddings_handle:
                self.embeddings, self.n_out = self.parse_embeddings(
                    embeddings_handle, lowercase=lowercase)
        elif isinstance(embeddings, file):
            self.embeddings, self.n_out = self.parse_embeddings(
                embeddings, lowercase=lowercase)
        elif isinstance(embeddings, dict):
            self.embeddings = embeddings
        else:
            raise TypeError('Invalid embeddings type: %s' % type(embeddings))

        self.size = len(self.embeddings)

        if not hasattr(id2word, '__getitem__'):
            raise TypeError('Supplied id2word mapping does NOT implement'
                            ' a __getitem__ method (type: %s).' % type(id2word))

        self.id2word = id2word
        self.op = op

        self.lowercase = lowercase
        self.dense = dense
        self.deUFALize = deUFALize

        self.oov = 0.0
        self.total_processed = 0.0
        self.oov_rate = 0.0

        self.hit_ids = set() # A set of word IDs that have been actually used.

        # DEACTIVATED
        self.log_oov_at = 1000  # Each log_oov_at calls to getitem, the OOV
                                # statistics are logged using logging.info()
        self.oov_collector = set()
        self.emptydocs = 0  # How many documents had no hit whatsoever

    def __getitem__(self, bow):
        """Transforms a given item from a list of words to an embedding vector.

        The vector is obtained using a max operation on each dimension of the
        embedding over the embeddings of individual words in the document.
        It is preferrable to call this method on a corpus and work with the
        resulting TransformedCorpus object's ``__getitem__`` method and/or
        iterate over it.

        :param bow: A sparse document vector (gensim-style) or a corpus. If
            a corpus is supplied, will return a :class:`TransformedCorpus`.

        :return: The document embedding or a transformed corpus that will
            output the document embeddings. Note that the returned vector
            is sparse, but can be set to dense using the ``dense`` parameter.
            In this way, the transformer will act like a dataset and return
            numpy arrays.
        """
        iscorp, corp = gensim.utils.is_corpus(bow)
        if iscorp is True:
            return self._apply(bow)

        if len(bow) == 0:
            logging.debug('Running empty doc through Word2VecTransformer.')
        else:
            #logging.debug('-- Word2VecTransformer doc length=%d --' % len(bow))
            pass

        embeddings = numpy.zeros((len(bow), self.n_out))
        has_hit = False
        for i, item in enumerate(bow):
            try:
                wid = item[0]
            except IndexError:
                logging.error('IndexError in item of type {0}: {1}'
                              ''.format(type(item), item))
                logging.error('Problem document: type {0}\n{1}'
                              ''.format(type(bow), bow))
                raise
            word = self._id2word(wid)
            try:
                embedding = self.embeddings[word]
                #logging.info('Embedding: %s with shape %s' % (
                #    btype(embedding), str(embedding.shape)))
                embeddings[i, :] = embedding
                has_hit = True
                self.hit_ids.add(wid)
                #  print '...hit.'
            except KeyError:
                self.oov += 1.0
                self.oov_collector.add((wid, word))
                #  print '...no hit.'
        if not has_hit:
            self.emptydocs += 1

        self.total_processed += len(bow)
        self.oov_rate = self.oov / self.total_processed

        # Combining the embeddings.
        output_embeddings = self.combine_words(embeddings)

        if self.dense:
            return output_embeddings

        sparse_embeddings = gensim.matutils.dense2vec(output_embeddings)
        return sparse_embeddings

    def _apply(self, corpus, chunksize=None):
        """Apply transformation in :func:`__getitem__` to the entire corpus.
        Does this by returning gensim's :class:`TransformedCorpus` object that
        applies the transformation over the entire corpus.

        :type corpus: gensim.interfaces.CorpusABC
        :param corpus: The corpus to transform.
        """
        # if not isinstance(corpus, TextCorpus):
        #     logging.warn('Word2VecTransformer applied on non-text' +
        #                  ' corpus; returning TransformedCorpus.')
        #
        #     transformed_corpus = TransformedCorpus(self, corpus, chunksize)
        #     return transformed_corpus
        # transformed_corpus = TransformedCorpus(self, corpus, chunksize)
        transformed_corpus = TransformedCorpus(self, corpus, chunksize)
        return transformed_corpus

    def _id2word(self, wid):
        """Converts the given word ID to a word, applying de-UFALization and
        lowercasing according to how the transformer was initialized."""
        word = self.id2word[wid]
        if self.lowercase:
            word = unicode(word).lower()
        if self.deUFALize:
            word = Word2VecTransformer.deUFALize(word)

        return word

    def combine_words(self, embeddings):
        """Computes a document representation from the word representations.

        :type embeddings: numpy.array
        :param embeddings: The embeddings matrix, one row per word.

        :rtype: numpy.array
        :return: The resulting vector.
        """
        if self.op == 'max':
            return numpy.max(embeddings, axis=0)
        elif self.op == 'sum':
            return numpy.sum(embeddings, axis=0)
        elif self.op == 'avg':
            return numpy.average(embeddings, axis=0)

    def log_oov(self):
        """Reports the OOV rate using the INFO logging level.
        """
        logging.info("Word2Vec OOV report: "
                     "total %d, oov %d, rate %f, unique %d, empty %d" % (
            int(self.total_processed), int(self.oov), self.oov_rate,
            len(self.oov_collector), int(self.emptydocs)))

    def report_oov(self):
        """Generates a report of queries that were in the id2word mapping but
        without an embedding."""
        oov_report = '\n'.join(["%d : %s" % (wid, word)
                                for (wid, word) in self.oov_collector])
        oov_report_header = '\nOOV report for word2vec\n' \
                              '=======================\n'
        return oov_report_header + oov_report

    def export_used(self):
        """Creates a Word2Vec transformer object reduced to words that have
        already been run through this instance of the transformer."""
        # Filter id2word dict
        id2word = { wid : self.id2word[wid] for wid in self.hit_ids }
        # Filter embeddings dict
        embeddings = { word : self.embeddings[word] for word in
                       map(self._id2word, id2word.keys()) }
        # create "spinoff" Word2VecTransformer
        new_transformer = Word2VecTransformer(embeddings=embeddings,
                                              id2word=id2word,
                                              op=self.op,
                                              lowercase=self.lowercase,
                                              dense=self.dense,
                                              deUFALize=self.deUFALize,
                                              from_pickle=False)
        return new_transformer

    def reset(self):
        """Resets itself to a "freshly initialized" state. Deletes all OOV
        and other usage statistics."""
        self.oov = 0.0
        self.total_processed = 0.0
        self.oov_rate = 0.0

        self.hit_ids = set() # A set of word IDs that have been actually used.

        # DEACTIVATED
        self.log_oov_at = 1000  # Each log_oov_at calls to getitem, the OOV
                                # statistics are logged using logging.info()
        self.oov_collector = set()
        self.emptydocs = 0  # How many documents had no hit whatsoever

    def __len__(self):
        """The length of the word2vec object is the vocabulary size."""
        return len(self.id2word)

    @staticmethod
    #@profile
    def parse_embeddings(input_handle, lowercase=False):
        """Builds the embeddings data structure.

        :param input_handle: The handle from which to read the dict.
            The expected format is one word per line,
            separated by whitespace or tabspace, with the first item being
            the word itself and the other items are the embedding vector.

            Additionally, the first line of the file contains two fields:
            the size of the vocabulary and the dimension of the embedding.

        :param lowercase: If set, will lowercase all keys.

        :rtype: dict(str => list(int)), int
        :returns: The embeddings dict and the embedding dimension.
        """
        header = next(input_handle)
        dict_size, n_out = itertools.imap(int, header.strip().split())

        e_dict = {}
        start_time = time.clock()
        for line in input_handle:
            fields = line.strip().split()
            word = fields[0]
            if lowercase:
                word = word.lower()
            embedding = numpy.array(list(itertools.imap(float, fields[1:])))
            # itertools.imap is about 2x faster than list comprehension here

            e_dict[word] = embedding

            if len(e_dict) % 10000 == 0:
                current_time = time.clock()
                time_taken = int(current_time - start_time)
                logging.info('Parsed %d entries, total time: %d s' % (
                    len(e_dict), time_taken)
                )

        logging.info('Parsing word2vec dict of %d entries took %d s.' % (
            len(e_dict), int(time.clock() - start_time))
        )

        if dict_size != len(e_dict):
            raise ValueError('Expected size of word2vec vocabulary' +
                             ' (%i)' % dict_size +
                             ' doesn\'t match actual size (%i).' % len(e_dict))

        return e_dict, n_out

    @staticmethod
    def deUFALize(word):
        """Removes lemmatization artifacts of Morphodita: numbers after
        dash (and the dash).

        :param word: The word to de-UFALize.

        :return: The raw lemma.
        """
        output = deUFALize_regex.sub('', word)
        return output
