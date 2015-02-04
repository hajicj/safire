"""
This module contains utility functions for working with TransformedCorpus
stacks. The functions in this module can look extremely dirty and whatnot; the
idea is to provide useful functionality OUTSIDE the core pipeline classes so
that they don't have to implement a complicated interface.

The most important function here is ``dimension()``. Other ones used in safire
are ``bottom_corpus()``, ``make_dense_output()``, ``id2word()`` and
``get_id2word_obj()``.

Of course, the downside is that if you write some special class yourself, you
will need to modify the functions here to work with that class.

When importing, try to stick to ``import safire.X`` and qualify types
(``y = safire.X.Y``) rather than ``from X import Y`` to avoid circular imports.
A lot of Safire classes depend on transcorp.py functions while at the same
time they need to be used inside transcorp.py to get access to their internals.
"""
import logging

from gensim.corpora import TextCorpus
from gensim.interfaces import TransformedCorpus
from gensim.models import TfidfModel
import gensim.matutils
import numpy

from safire.data import FrequencyBasedTransformer, VTextCorpus
import safire.data.serializer
from safire.data.sharded_corpus import ShardedCorpus
#from safire.datasets.dataset import Dataset
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.word2vec_transformer import Word2VecTransformer
#from safire.datasets.transformations import DatasetTransformer
import safire.datasets.dataset
#import safire.datasets.transformations
from safire.utils import IndexedTransformedCorpus
import safire.utils.transformers


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
    # TODO: Move this mechanism into transformers themselves?
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
    """Jumps through a stack of TransformedCorpus or Dataset
    objects all the way to the bottom corpus."""
    current_corpus = corpus
    if isinstance(current_corpus, TransformedCorpus):
        return bottom_corpus(current_corpus.corpus)
    if isinstance(current_corpus, safire.datasets.dataset.DatasetABC):
        return bottom_corpus(current_corpus.data)
    return current_corpus


def dimension(corpus):
    """Finds the topmost corpus that can provide information about its
    output dimension."""
    if isinstance(corpus, numpy.ndarray) and len(corpus.shape) == 2:
        return corpus.shape[1]
    current_corpus = corpus
    if hasattr(current_corpus, 'dim'):
        return current_corpus.dim
    if hasattr(current_corpus, 'n_out'):
        return current_corpus.n_out  # This is stupid! It's an *output* dimension.
    if hasattr(current_corpus, 'n_in'):
        return current_corpus.n_in  # This is stupid! It's an *output* dimension.
    if isinstance(current_corpus, TextCorpus):
        return len(current_corpus.dictionary)
    if isinstance(current_corpus, ImagenetCorpus):
        return current_corpus.dim
    if isinstance(current_corpus, ShardedCorpus):
        return current_corpus.dim

    # This is the "magic". There's no unified mechanism for providing output
    # corpus dimension in gensim, so we have to deal with it case-by-case.
    # Stuff that is in safire usually has an 'n_out' attribute (or, in other
    # words: whenever something in safire has the 'n_out' attribute, it means
    # the output dimension).
    if isinstance(current_corpus, TransformedCorpus):
        if isinstance(current_corpus.obj, FrequencyBasedTransformer):
            # This should change to something more consistent (dictionary size?)
            return current_corpus.obj.k - current_corpus.obj.discard_top
        # Optimization, TfidfModel doesn't change model dimension
        elif isinstance(current_corpus.obj, TfidfModel):
            if hasattr(current_corpus.obj, 'dfs'):
                return len(current_corpus.obj.dfs)
        elif hasattr(current_corpus.obj, 'dim'): # Covers SafireTransformers
            return current_corpus.obj.dim
        elif hasattr(current_corpus.obj, 'n_out'): # Covers SafireTransformers
            return current_corpus.obj.n_out
        else:
            return dimension(current_corpus.corpus)

    else:
        raise ValueError('Cannot find output dimension of corpus %s' % str(corpus))


def get_composite_source(pipeline, name):
    """Retrieves the pipeline of a named data source from a composite dataset
    down the (presumably composite) pipeline."""
    if not isinstance(name, str):
        raise TypeError('Composite source dimension can only be derived for'
                        'data source names, you supplied {0} of type {1}.'
                        ''.format(name, type(name)))
    if isinstance(pipeline, safire.datasets.dataset.CompositeDataset):
        return pipeline[name]
    else:
        if isinstance(pipeline, TransformedCorpus):
            return get_composite_source(pipeline.corpus, name)
        elif isinstance(pipeline, safire.datasets.dataset.DatasetABC):
            return get_composite_source(pipeline.data, name)
        else:
            raise TypeError('Cannot derive composite source dimension from'
                            'a non-pipeline object (type: {0}). Are you sure'
                            'your pipeline had a CompositeDataset block?'
                            ''.format(type(pipeline)))


def run_transformations(item, *transformations):
    """Runs the TransformedCorpus transformation stack."""
    out = item
    for tr in transformations:
        out = tr[out]
        print 'Transformation applied: %s' % str(tr)
        print 'Result: %s' % str(out)
    return out


# def get_transformers(corpus):
#     """Recovers the Transformation objects from a stack of TransformedCorpora.
#     """
#     tr = []
#     current_corpus = corpus
#     while isinstance(current_corpus, TransformedCorpus):
#         tr.append(current_corpus.obj)
#         current_corpus = current_corpus.corpus
#     if isinstance(current_corpus, VTextCorpus): # Also has __getitem__...
#         tr.append(current_corpus)
#     tr.reverse()
#     return tr

def get_transformers(pipeline):
    """Recovers the Transformation objects from a pipeline.

    Problems handling Swapout: the recovered list of transformers
    should be enough to re-run the pipeline, but there's a problem with gensim
    to dense conversion within a SwapoutCorpus that delegates retrieval to a
    ShardedCorpus, thus performing gensim->dense conversion silently, without
    an intervening block.

    This problem should NOT be handled inside
    ShardedCorpus, because it's not something ShardedCorpus is for.

    Can it be handled by Serializer transformer? Instead of hooking the
    ShardedCorpus directly to SwapoutCorpus, it could hook *itself* and call
    its self.corpus.__getitem__ on a SwapoutCorpus.__getitem__ call -- and if
    a transformation is silently happening there, on a transformer call, should
    just cast the gensim vector(s) to ndarray using gensim2ndarray().

    Problem handling Datasets: since the Dataset just passes the __getitem__
    call to its underlying corpus, it acts as a transformer and corpus in one.
    """
    if isinstance(pipeline, safire.data.serializer.SwapoutCorpus):
        logging.warn('get_transformers(): Corpus of type {0}, skipping '
                     'transformer.'.format(type(pipeline)))
        # ...and here, ladies and gents, we show how terrible this module is.

        # If a silent conversion is happening...
        if isinstance(pipeline.obj, ShardedCorpus) and \
                        pipeline.obj.gensim is False and \
                        pipeline.obj.sparse_retrieval is False:
            # ..add a convertor, so that the resulting stack of transformers
            # can be run.
            return get_transformers(pipeline.corpus) + \
                   [safire.utils.transformers.Corpus2Dense(
                       dim=pipeline.obj.dim)]

        # This will break on the ShardedCorpus being set to sparse_retrieval,
        # but who cares.
        return get_transformers(pipeline.corpus)
    if isinstance(pipeline, TransformedCorpus):
        return get_transformers(pipeline.corpus) + [pipeline.obj]
    if isinstance(pipeline, safire.datasets.dataset.TransformedDataset):
        return get_transformers(pipeline.data) + [pipeline.obj]
    # Ignoring cast to dataset???
    if isinstance(pipeline, safire.datasets.dataset.DatasetABC):
        logging.warn('get_transformers(): Corpus of type {0}, using as '
                     'transformer.'.format(type(pipeline)))
        return get_transformers(pipeline.data) #+ [pipeline]
    return [] # If the bottom has been reached.


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

    def __len__(self):
        return len(self.keymap)


def log_corpus_stack(corpus):
    """Reports the types of corpora and transformations of a given
    corpus stack. Currently cannot deal with CompositeDataset pipelines."""
    if isinstance(corpus, TransformedCorpus):
        r = 'Type: {0} with obj {1}'.format(type(corpus), type(corpus.obj))
        return '\n'.join([r, log_corpus_stack(corpus.corpus)])
    elif isinstance(corpus, safire.datasets.dataset.TransformedDataset):
        r = 'Type: %s with obj %s' % (type(corpus), type(corpus.obj))
        return '\n'.join([r, log_corpus_stack(corpus.data)])
    elif isinstance(corpus, safire.datasets.dataset.CompositeDataset):
        r = 'Type: {0} with the following datasets: {1}'.format(
            type(corpus),
            ''.join(['\n    {0}'.format(type(d)) for d in corpus.data])
        )
        individual_logs = [log_corpus_stack(d) for d in corpus.data]
        combined_logs = '------component-------\n' + \
                        '------component-------\n'.join(individual_logs)
        return '\n'.join([r, combined_logs])
    elif isinstance(corpus, safire.datasets.dataset.DatasetABC):
        r = 'Type: {0}, passing through DatasetABC to underlying corpus {1}' \
            ''.format(type(corpus), type(corpus.data))
        return '\n'.join([r, log_corpus_stack(corpus.data)])
    else:
        r = 'Type: %s' % (type(corpus))
        return '\n'.join([r, '=== STACK END ===\n'])


def convert_to_dense(corpus):
    """Adds a utility block that outputs items in a dense format.

    If the given corpus is of a type that can support dense output by itself
    (for example a SwapoutCorpus with a ShardedCorpus back-end), will instead
    set the corpus output type to dense.

    This function is called by DatasetABC on initialization if the
    ``ensure_dense`` option is set.
    """
    if isinstance(corpus, safire.data.serializer.SwapoutCorpus) \
        and isinstance(corpus.obj, ShardedCorpus):
            corpus.obj.gensim = False
            corpus.obj.sparse_retrieval = False
            return corpus
    elif isinstance(corpus, safire.datasets.dataset.DatasetABC) \
            and isinstance(corpus[0], numpy.ndarray):
        logging.info('DatasetABC has dense output by default (this very method'
                     'gets called during initialization).')
        return corpus
    elif isinstance(corpus, IndexedTransformedCorpus) \
            and isinstance(corpus[0], numpy.ndarray):
        logging.warn('Corpus class {0}: conversion to dense already done'
                     ' downstream somewhere, no change.'.format(type(corpus)))
        return corpus
    else:
        logging.warn('Corpus class {0}: cannot rely on '
                     'ShardedCorpus.gensim=False, assuming gensim sparse '
                     'vector output and applying Corpus2Dense.'
                     ''.format(type(corpus)))

        transformer = safire.utils.transformers.Corpus2Dense(corpus)
        # Have to _apply to make sure the output is a pipeline, because
        # Corpus2Dense call on __getitem__ might call gensim2dense directly
        # on something that behaves like a corpus but is not an instance of
        # CorpusABC (like: Datasets? but why would we want to ensure dense
        # output on Datasets like this?)
        return transformer._apply(corpus)


def find_type_in_pipeline(pipeline, type_to_find):
    """Finds the topmost instance of the given block type in the given pipeline.
    Returns the given block."""
    if isinstance(pipeline, type_to_find):
        return pipeline
    else:
        return find_type_in_pipeline(pipeline.corpus)


def convert_to_dense_recursive(pipeline):
    """Whenever possible, sets transformation inputs/outputs to be numpy
    ndarrays, to do away with gensim2ndarray/ndarray2gensim conversions at
    block boundaries.

    This should be possible if:

    * there is a corpus in the pipeline that can supply dense data
      (numpy ndarrays, normally a SwapoutCorpus with a ShardedCorpus obj),
    * all transformers from that point on can work directly on numpy ndarrays,
    * all TransformedCorpus objects on the pipeline do not interfere with the
      type of the data they pass on during __getitem__ or __iter__ calls
      (this may be a problem for __iter__ when chunksize is set).
    """
    #
    # WIP, do not use.
    raise NotImplementedError()

    # Find the last SwapoutCorpus that can give us dense data.
    # Set this corpus to dense output.
    # Set all remaining transformers to dense throughput.
    def _todense_recursive(corpus):
        # This is so far the only supported case of converting to dense.
        if isinstance(corpus, safire.data.serializer.SwapoutCorpus):
            if isinstance(corpus.obj, safire.data.sharded_corpus.ShardedCorpus):
                corpus.obj.gensim = False
                corpus.obj.sparse_retrieval = False
                return
            else:
                # Let's try whether we can make the obj from which SwapoutCorpus
                # is retrieving also dense.
                _todense_recursive(corpus.obj)

        # End of pipeline: ensure dense, insert Corpus2Dense?
        if not isinstance(corpus, TransformedCorpus):
            pass


        # if has_dense_throughput(corpus):
        #     _todense_recursive(corpus)
        #     set_dense_throughput(corpus)


def smart_apply_transcorp(obj, corpus, *args, **kwargs):
    """Used inside a transformer's _apply() method.
    Decides whether to initialize a TransformedCorpus, or an
    IndexedTransformedCorpus."""
    try:
        return IndexedTransformedCorpus(obj, corpus, *args, **kwargs)
    except TypeError:
        return TransformedCorpus(obj, corpus, *args, **kwargs)