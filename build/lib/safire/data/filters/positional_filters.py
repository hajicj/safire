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
    """Retain words that occur in the first K sentences (but retain all their
    occurrences in the following sentences as well)."""
    retain_words = frozenset(list(itertools.chain(*sentences[:k])))
    out_sentences = [[w for w in s if w in retain_words] for s in sentences]
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
    out_sentences = [[w for w in s if w in retain_words] for s in sentences]
    return out_sentences


def first_k_plus_p(sentences, k, p):
    """Returns the first ``k + (n_sentences - k) * p`` sentences."""
    k_sentences = first_k(sentences, k)
    p_sentences = first_p(sentences[k:], p)
    return list(itertools.chain(k_sentences, p_sentences))


def words_from_first_k_plus_p(sentences, k, p):
    """Returns words from the first ``k + (n_sentences - k) * p`` sentences."""
    k_plus_p_sentences = first_k_plus_p(sentences, k, p)
    words = frozenset(itertools.chain(*k_plus_p_sentences))
    output = [[w for w in s if w in words] for s in sentences]
    return output