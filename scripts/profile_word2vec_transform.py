#!/usr/bin/env python
"""
``profile_word2vec_transform.py`` is a script that measures performance of the
Word2VecTransformer class. This is really just a small helper script.
"""
import argparse
import codecs
import logging
import os
import random
import sys

from safire.utils import profile_run, total_size
from safire.data.word2vec_transformer import Word2VecTransformer

__author__ = 'Jan Hajic jr.'


default_embeddings_dir = os.path.join('c:\\Users', 'Lenovo', 'word2vec')
default_vectors_fname = 'ces_head.vectors'
default_vocab_fname = 'ces_head.vocab'
default_embeddings = os.path.join(default_embeddings_dir,
                                  default_vectors_fname)
default_vocab = os.path.join(default_embeddings_dir,
                             default_vocab_fname)

##############################################################################

def init_dummy_id2word(vocab_file):
    """Initializes a dummy id2word dict based on the vocab file.

    :param vocab_file: A word2vec vocabulary filename.

    :return: A dictionary with integer keys and word values.
    """
    id2word = {}
    with codecs.open(vocab_file, 'r', 'utf-8') as vhandle:
        for i, line in enumerate(vhandle):
            word, _ = line.strip().split()
            id2word[i] = word
    return id2word


def init_w2v(filename, id2word, **kwargs):
    """Initializes the Word2VecTransformer.

    :param filename: The embeddings filename.

    :param id2word: An id2word dict. Use the dummy dict from
        :func:`init_dummy_id2word`.

    :return: The Word2Vec transformer object.
    """
    w2v = Word2VecTransformer(filename, id2word, **kwargs)
    return w2v


#############

# Generating mock data

def generate_dummy_vector(range=10000, length_range=10):
    """Generates a dummy sparse vector with up to ``length_range`` entries
    from 0 to ``range``."""
    output = [ (random.randint(0, range-1), random.uniform(0, 2))
               for _ in xrange(random.randint(1, length_range)) ]
    return output


def generate_dummy_vectors(n_vectors=1000, range=10000, length_range=10):
    return [generate_dummy_vector(range=range, length_range=length_range)
            for _ in xrange(n_vectors)]


def w2v_getitem_bench(word2vec, n_calls=10000):

    queries = generate_dummy_vectors(n_calls, range=len(word2vec))

    for query in queries:
        output = word2vec[query]

    pass

#############################################################################


def main(args):
    logging.info('Executing profile_word2vec_transform.py...')

    id2word_report, id2word = profile_run(init_dummy_id2word, args.vocab)
    print
    print 'Parsing the vocab file, creating a dummy id2word dict'
    print '====================================================='
    print id2word_report.getvalue()

    print 'id2word total size: %d' % total_size(id2word)

    w2v_init_report, word2vec = profile_run(init_w2v,
                                            args.embeddings,
                                            id2word,
                                            lowercase=False)
    print
    print 'Initializing the Word2VecTransformer'
    print '===================================='
    print w2v_init_report.getvalue()

    #print 'word2vec total size: %d' % total_size(word2vec)

    # Randomly calling getitem()
    n_calls = 10000
    getitem_report, _ = profile_run(w2v_getitem_bench, word2vec, n_calls)
    print
    print 'Benchmarking __getitem__'
    print '========================'
    print getitem_report.getvalue()



    logging.info('Exiting profile_word2vec_transform.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    parser.add_argument('-e', '--embeddings', action='store',
                        default=default_embeddings,
                        help='Path to the vectors file that contains word2vec'
                             ' embeddings. Expects one entry per line, delim.'
                             ' by something split()-able. First field is word,'
                             ' other fields are embedding entries.')
    parser.add_argument('--vocab', action='store',
                        default=default_vocab,
                        help='Path to the vocab file that contains word2vec'
                             ' words and their frequencies. Expects one entry'
                             'per line, delim. by something split()-able.'
                             'First field is word, second field is frequency.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO logging messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG logging messages. (May get very '
                             'verbose.')

    return parser


if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    main(args)
