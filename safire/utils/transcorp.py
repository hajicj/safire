"""
This module contains utility functions for working with gensim TransformedCorpus
stacks.
"""
import logging

from gensim.corpora import TextCorpus
from gensim.interfaces import TransformedCorpus
from gensim.models import TfidfModel

from safire.data import FrequencyBasedTransformer, VTextCorpus
from safire.datasets.dataset import Dataset
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.word2vec_transformer import Word2VecTransformer


__author__ = "Jan Hajic jr."


def id2word(corpus, wid):
    """Returns the word associated with the original ID in the given corpus.
    Based on corpus type, this can mean backtracking the origin of the ID
    number through multiple transformations.

    Currently implemented: backtracking through
    :class:`FrequencyBasedTransformer`.
    """

    if isinstance(corpus, TransformedCorpus):
        if isinstance(corpus.obj, FrequencyBasedTransformer):
            orig_wid = corpus.obj.transformed2orig[wid]
            logging.debug('Running id2word through FrequencyBasedTransformer: from %d to %d' % (wid, orig_wid))
            return unicode(id2word(corpus.corpus, orig_wid))
        else:
            #print 'Running id2word through TransformedCorpus: staying at %d' % wid
            return unicode(id2word(corpus.corpus, wid))
    elif hasattr(corpus, 'dictionary'):
        logging.debug('Found corpus with dictionary, wid: %d' % wid)
        return unicode(corpus.dictionary[wid])
    else:
        raise ValueError('Cannot backtrack through corpus type %s' % str(
            type(corpus)))


def get_id2word_obj(corpus):
    """Retrieves the valid id2word object that can handle ``__getitem__``
    requests on word IDs to return the words themselves."""
    if isinstance(corpus, VTextCorpus):
        return corpus.dictionary
    elif isinstance(corpus, TransformedCorpus):
        if isinstance(corpus.obj, FrequencyBasedTransformer):
            return KeymapDict(get_id2word_obj(corpus.corpus),
                              corpus.obj.transformed2orig)
        elif isinstance(corpus.obj, Word2VecTransformer):
            return corpus.obj.id2word
        else:
            return get_id2word_obj(corpus.corpus)

    raise NotImplementedError()


def bottom_corpus(corpus):
    """Jumps through a stack of TransformedCorpus objects all the way to the
    bottom corpus."""
    current_corpus = corpus
    while isinstance(current_corpus, TransformedCorpus):
        current_corpus = current_corpus.corpus
    return current_corpus


def dimension(corpus):
    """Finds the topomost corpus that can provide information about its
    output dimension."""
    current_corpus = corpus
    if isinstance(current_corpus, Dataset):
        return current_corpus.n_in
    if isinstance(current_corpus, TextCorpus):
        return len(current_corpus.dictionary)
    if isinstance(current_corpus, ImagenetCorpus):
        return current_corpus.dim

    # This is the "magic". There's no unified mechanism for providing output
    # corpus dimension in gensim, so we have to deal with it case-by-case.
    # Stuff that is in safire usually has an 'n_out' attribute (or, in other
    # words: whenever something in safire has the 'n_out' attribute, it means
    # the output dimension).
    if isinstance(current_corpus, TransformedCorpus):
        if isinstance(current_corpus.obj, FrequencyBasedTransformer):
            return current_corpus.obj.k - current_corpus.obj.discard_top
        # Optimization, TfidfModel doesn't change model dimension
        elif isinstance(current_corpus.obj, TfidfModel):
            if hasattr(current_corpus.obj, 'dfs'):
                return len(current_corpus.obj.dfs)
        elif hasattr(current_corpus.obj, 'n_out'): # Covers SafireTransformers
            return current_corpus.obj.n_out
        else:
            return dimension(current_corpus.corpus)
    else:
        raise ValueError('Cannot find output dimension of corpus %s' % str(corpus))


def run_transformations(item, *transformations):
    """Runs the TransformedCorpus transformation stack."""
    out = item
    for tr in transformations:
        #print 'Transformation applied: %s' % str(tr)
        out = tr[out]
        #print 'Result: %s' % str(out)
    return out


def get_transformers(corpus):
    """Recovers the Transformation objects from a stack of TransformedCorpora.
    """
    tr = []
    current_corpus = corpus
    while isinstance(current_corpus, TransformedCorpus):
        tr.append(current_corpus.obj)
        current_corpus = current_corpus.corpus
    if isinstance(current_corpus, VTextCorpus): # Also has __getitem__...
        tr.append(current_corpus)
    tr.reverse()
    return tr


def reset_vtcorp_input(corpus, filename, input_root=None, lock=True,
                       inplace=True):
    """Resets the inputs for the VTextCorpus at the bottom of the
    TransformedCorpus stack.

    :param inplace: If this flag is set, will switch the inputs for the given
        corpus in-place. If not, will deepcopy the corpus. **[NOT IMPLEMENTED]
        Don't use (stick with True)!**"""
    vtcorp = bottom_corpus(corpus)
    if not isinstance(vtcorp, VTextCorpus):
        raise ValueError('Bottom corpus '
                         '%s instead of VTextCorpus.' % type(vtcorp))
    vtcorp.reset_input(filename, input_root=input_root, lock=lock)


class KeymapDict(object):
    """Implements a dict wrapped in a key mapping."""
    def __init__(self, dict, keymap):
        self.dict = dict
        self.keymap = keymap

    def __getitem__(self, item):
        return self.dict[self.keymap[item]]


def log_corpus_stack(corpus):
    """Reports the types of corpora and transformations of a given
    corpus stack."""
    if isinstance(corpus, TransformedCorpus):
        r = 'Type: %s with obj %s' % (type(corpus), type(corpus.obj))
        return '\n'.join([r, log_corpus_stack(corpus.corpus)])
    else:
        r = 'Type: %s' % (type(corpus))
        return '\n'.join([r, '=== STACK END ===\n'])