"""
This module contains classes that ...
"""
import codecs
import logging
import numpy
import time

import gensim
from gensim.corpora import TextCorpus
from gensim.interfaces import TransformationABC, TransformedCorpus

__author__ = "Jan Hajic jr."


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

    def __init__(self, embeddings, id2word):
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
        """
        if isinstance(embeddings, str):
            with codecs.open(embeddings, 'r', 'utf-8') as embeddings_handle:
                self.embeddings, self.n_out = self.parse_embeddings(
                    embeddings_handle)
        else:
            self.embeddings, self.n_out = self.parse_embeddings(embeddings)

        self.size = len(self.embeddings)

        if not hasattr(id2word, '__getitem__'):
            raise TypeError('Supplied id2word mapping does NOT implement'
                            ' a __getitem__ method (type: %s).' % type(id2word))

        self.id2word = id2word  # TODO: add processing such as lowercasing
        self._id2word_fn = self.id2word.__getitem__

    def __getitem__(self, bow, dense=False):
        """Transforms a given item from a list of words to an embedding vector.

        The vector is obtained using a max operation on each dimension of the
        embedding over the embeddings of individual words in the document.
        It is preferrable to call this method on a corpus and work with the
        resulting TransformedCorpus object's ``__getitem__`` method and/or
        iterate over it.

        :param bow: A sparse document vector (gensim-style) or a corpus. If
            a corpus is supplied, will return a :class:`TransformedCorpus`.

        :param dense: If set, will return a dense embedding instead of
            sparsifying it.

        :return: The document embedding or a transformed corpus that will
            output the document embeddings. Note that the returned vector
            is sparse, but can be set to dense using the ``dense`` parameter.
            In this way, the transformer will act like a dataset and return
            numpy arrays.
        """
        iscorp, corp = gensim.utils.is_corpus(bow)
        if iscorp is True:
            return self._apply(bow, dense=dense)

        embeddings = numpy.zeros((len(bow), self.n_out))
        for i, item in enumerate(bow):
            wid = item[0]
            word = self.id2word[wid]
            try:
                embedding = self.embeddings[word]
                embeddings[i] = embedding
            except KeyError:
                pass

        # Combining the embeddings. (Could be a method.)
        maxout_embeddings = numpy.max(embeddings, axis=0)

        if dense:
            return maxout_embeddings

        sparse_embeddings = gensim.matutils.dense2vec(maxout_embeddings)
        return sparse_embeddings

    def _apply(self, corpus, chunksize=None):
        """Apply transformation in :func:`__getitem__` to the entire corpus.
        Does this by returning gensim's :class:`TransformedCorpus` object that
        applies the transformation over the entire corpus.

        :type corpus: gensim.interfaces.CorpusABC
        :param corpus: The corpus to transform.
        """
        if not isinstance(corpus, TextCorpus):
            logging.warn('Word2VecTransformer applied on non-text' +
                         ' corpus; returning TransformedCorpus.')

            transformed_corpus = TransformedCorpus(self, corpus, chunksize)
            return transformed_corpus

        transformed_corpus = TransformedCorpus(self, corpus, chunksize)
        return transformed_corpus


    @staticmethod
    def parse_embeddings(input_handle):
        """Builds the embeddings data structure.

        :param input_handle: The handle from which to read the dict.
            The expected format is one word per line,
            separated by whitespace or tabspace, with the first item being
            the word itself and the other items are the embedding vector.

            Additionally, the first line of the file contains two fields:
            the size of the vocabulary and the dimension of the embedding.

        :rtype: dict(str => list(int)), int
        :returns: The embeddings dict and the embedding dimension.
        """
        header = next(input_handle)
        dict_size, n_out = map(int, header.strip().split())

        e_dict = {}
        start_time = time.clock()
        for line in input_handle:
            fields = line.strip().split()
            word = fields[0]
            embedding = fields[1:]

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

