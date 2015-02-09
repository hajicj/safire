"""
Implements functions that filter out certain words or sentences from an array
of sentences parsed from a vertical text file.

Each function here takes as the first argument the sentences array and some
optional keyword arguments.
"""
import itertools
import math

__author__ = 'Jan Hajic jr.'



def first_k(sentences, k=None):
    """Retain only the first K sentences."""
    return sentences[:k]


def words_from_first_k(sentences, k):
    """Retain only words that occur in the first K sentences."""
    retain_words = frozenset(list(itertools.chain(*sentences[:k])))
    out_sentences = [ [ w for w in s if w in retain_words ] for s in sentences ]
    #print "sentences: %s" % '\n'.join([' '.join(s) for s in sentences])
    #print "out_sentences: %s" % '\n'.join([' '.join(s) for s in out_sentences])
    return out_sentences


def first_p(sentences, p):
    """Retain only the given fraction of sentences from the beginning.
    (Rounds up, for single-sentence documents.)"""
    k = int(math.ceil(len(sentences) * p))
    return sentences[:k]


def words_from_first_p(sentences, p):
    """Retain only words that occur in the given fraction of sentences from
    the beginning."""
    k = int(math.ceil(len(sentences) * p))
    retain_words = frozenset(list(itertools.chain(*sentences[:k])))
    out_sentences = [ [ w for w in s if w in retain_words ] for s in sentences ]
    return out_sentences
