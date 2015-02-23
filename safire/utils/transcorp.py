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
import collections
import itertools
import logging
import copy

from gensim.corpora import TextCorpus
from gensim.interfaces import TransformedCorpus
from gensim.models import TfidfModel
import gensim.matutils
import numpy

from safire.data import FrequencyBasedTransformer, VTextCorpus
import safire.data.serializer
import safire.data.sharded_corpus
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.data.sharded_corpus import ShardedCorpus
from safire.data.word2vec_transformer import Word2VecTransformer
import safire.datasets.dataset
from safire.utils import IndexedTransformedCorpus, freqdict
import safire.utils
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
    elif isinstance(corpus, safire.datasets.dataset.DatasetABC):
        return get_id2word_obj(corpus.data)

    raise NotImplementedError('get_id2word_obj() not implemented for corpus '
                              'type {0}'.format(type(corpus)))


def get_id2doc_obj(corpus):
    if hasattr(corpus, 'id2doc'):
        return corpus.id2doc
    elif isinstance(corpus, TransformedCorpus):
        return get_id2doc_obj(corpus.corpus)
    elif isinstance(corpus, safire.datasets.dataset.DatasetABC):
        return get_id2doc_obj(corpus.data)

    raise NotImplementedError('get_id2doc_obj() not implemented for corpus '
                              'type {0}'.format(type(corpus)))


def id2doc(corpus, docid):
    obj = get_id2doc_obj(corpus)
    return obj[docid]


def get_doc2id_obj(corpus):
    if hasattr(corpus, 'doc2id'):
        # print 'Returning doc2id from corpus: {0}'.format(type(corpus))
        return corpus.doc2id
    elif isinstance(corpus, TransformedCorpus):
        return get_doc2id_obj(corpus.corpus)
    elif isinstance(corpus, safire.datasets.dataset.DatasetABC):
        return get_doc2id_obj(corpus.data)

    raise NotImplementedError('get_doc2id_obj() not implemented for corpus '
                              'type {0}'.format(type(corpus)))


def doc2id(corpus, docname):
    obj = get_doc2id_obj(corpus)
    return obj[docname]


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
    # print 'Deriving dimension of corpus {0}'.format(type(corpus))
    if isinstance(corpus, numpy.ndarray) and len(corpus.shape) == 2:
        return corpus.shape[1]
    current_corpus = corpus
    if hasattr(current_corpus, 'dim'):
        # Covers almost everything non-recursive in safire.
        return current_corpus.dim
    if hasattr(current_corpus, 'n_out'):
        return current_corpus.n_out
    if hasattr(current_corpus, 'n_in'):
        return current_corpus.n_in  # This is stupid! It's *output* dimension.
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
        # print 'Transformation applied: %s' % str(tr)
        # print 'Result: %s' % str(out)
    return out


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
                        pipeline.obj.gensim_retrieval is False and \
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
                         '{0} instead of VTextCorpus.'.format(type(vtcorp)))
    vtcorp.reset_input(filename, input_root=input_root, lock=lock)


class KeymapDict(gensim.corpora.Dictionary):
    """Implements a dict wrapped in a key mapping."""
    def __init__(self, dict, keymap):
        self.dict = dict
        self.keymap = keymap

    def __getitem__(self, item):
        return self.dict[self.keymap[item]]

    def __len__(self):
        return len(self.keymap)


def keymap2dict(keymap_dict):
    """Builds a new id2word dictionary so that only the items that are in
    keymap are retained and they get the keymapped IDs.

    If the original ID of token ``X`` is ``1`` and the keymapped ID is ``4``,

    It relies on gensim's Dictionary internals, so this is not a very safe
    function. However, probability of changes to this part of gensim are pretty
    low.

    :type keymap_dict: KeymapDict
    :param keymap_dict: A KeymapDict which to use to generate a new fully fledged
        gensim Dictionary.

    :return: The new dictionary.
    """
    old_dict = keymap_dict.dict
    id_old2new = {v: k for k, v in keymap_dict.keymap.iteritems()}

    logging.debug('Old dict type: {0}'.format(type(old_dict)))
    logging.debug('id_old2new: {0}'.format(id_old2new))

    # The dictionary needs to be built manually.
    new_dict = gensim.corpora.Dictionary()

    for token, old_id in old_dict.token2id.items():
        logging.debug(u'Token: {0}, old ID: {1}'.format(token, old_id))
        if old_id not in id_old2new:
            logging.debug(u'    Filtering out token "{0}"'.format(token))
            continue
        new_token_id = id_old2new[old_dict.token2id[token]]
        new_dict.token2id[token] = new_token_id
        new_dict.id2token[new_token_id] = token

        # Also copy document frequency information
        new_dict.dfs[new_token_id] = old_dict.dfs[old_id]

    new_dict.num_docs = old_dict.num_docs
    new_dict.num_pos = old_dict.num_pos
    # This number should reflect only corpus positions with the filtered
    # vocabulary, but that's impossible to do without the original frequency
    # filtering object, as gensim's Dictionary does not keep frequency data.
    new_dict.num_nnz = old_dict.num_nnz

    return new_dict


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
    # Straightforward: if we have a dense output-capable corpus, set it to
    # dense output.
    if isinstance(corpus, safire.data.serializer.SwapoutCorpus) \
            and isinstance(corpus.obj, ShardedCorpus):
            corpus.obj.gensim_retrieval = False
            corpus.obj.sparse_retrieval = False
            return corpus

    # If we have a dataset that already is outputting dense data, leave it be.
    elif isinstance(corpus, safire.datasets.dataset.DatasetABC) \
            and isinstance(corpus[0], numpy.ndarray):
        logging.info('Dataset class {0}: conversion to dense already done'
                     ' downstream somewhere, no change.'.format(type(corpus)))
        return corpus

    # If we have an IndexedTransformedCorpus that is already outputting dense
    # data, leave it be.
    elif isinstance(corpus, IndexedTransformedCorpus) \
            and isinstance(corpus[0], numpy.ndarray):
        logging.warn('Corpus class {0}: conversion to dense already done'
                     ' downstream somewhere, no change.'.format(type(corpus)))
        return corpus

    elif isinstance(corpus, TransformedCorpus) and \
        isinstance(corpus.obj, safire.utils.transformers.Corpus2Dense):
        logging.info('Corpus class {0}: last transformer already is '
                     'Corpus2Dense.')
        return corpus

    # In case we cannot guarantee dense output:
    else:
        logging.warn('Corpus class {0}: cannot rely on pre-existing dense '
                     'output or ShardedCorpus.gensim=False, assuming gensim '
                     'sparse vector output and applying Corpus2Dense.'
                     ''.format(type(corpus)))

        transformer = safire.utils.transformers.Corpus2Dense(corpus)
        # Have to _apply to make sure the output is a pipeline, because
        # Corpus2Dense call on __getitem__ might call gensim2dense directly
        # on something that behaves like a corpus but is not an instance of
        # CorpusABC.
        return transformer._apply(corpus)


def find_type_in_pipeline(pipeline, type_to_find):
    """Finds the topmost instance of the given block type in the given pipeline.
    Returns the given block. If the given type is not found, returns None."""
    if isinstance(pipeline, type_to_find):
        return pipeline
    elif isinstance(pipeline, TransformedCorpus):
        return find_type_in_pipeline(pipeline.corpus, type_to_find)
    elif isinstance(pipeline, safire.datasets.dataset.DatasetABC):
        return find_type_in_pipeline(pipeline.data, type_to_find)
    else:
        return None


def is_serialized(pipeline, serializer_class=ShardedCorpus):
    """Checks if the pipeline contains a serializer that used the given class
    for serialization."""
    swapout = find_type_in_pipeline(pipeline,
                                    safire.data.serializer.SwapoutCorpus)
    if swapout is None:
        return False
    else:
        return (isinstance(swapout.obj, safire.data.serializer.Serializer)
                and isinstance(swapout.obj.serializer_class, serializer_class))


def is_fully_indexable(pipeline):
    """Checks whether the pipeline is indexable, i.e. whether it responds to
    __getitem__ requests. Presupposes that the pipeline has at least one
    data point.

    Checks only duck typing.
    """
    if len(pipeline) == 0:
        raise ValueError('Cannot inspect empty pipeline!')
    try:
        _ = pipeline[0]
        _ = pipeline[0:1]
        _ = pipeline[:1]
        _ = pipeline[-1:]
        _ = pipeline[[0]]
        return True
    except (TypeError, AttributeError, ValueError):
        raise
        return False


def ensure_serialization(pipeline, force=False, serializer_class=ShardedCorpus,
                         **serializer_kwargs):
    """Checks if the pipeline has been serialized using the given class.
    If not, serializes the class using the supplied kwargs.

    This is used when you need to make sure that all information about the
    pipeline is available for processing further down the line, like when
    flattening with another pipeline.

    The kwargs typically have to contain the ``fname`` argument, to tell the
    serializer class where the data should go.

    :param force: If this flag is set, will serialize not just if the pipeline
        has a serialization block somewhere; it will serialize unless the top
        block is a serialized SwapoutCorpus.

    :returns: The original pipeline if it already has been serialized,
        otherwise it returns the pipeline with a serializer block on top."""
    reserialize = False
    if force and not (isinstance(pipeline, safire.data.serializer.SwapoutCorpus)
                      and isinstance(pipeline.obj,
                                     safire.data.serializer.Serializer)
                      and isinstance(pipeline.obj.serializer_class,
                                     serializer_class)):
        reserialize = True
    if not reserialize and not is_serialized(pipeline, serializer_class):
        reserialize = True
    if reserialize:
        logging.info('Pipeline {0} not serialized, serializing using class {1}'
                     ''.format(pipeline, serializer_class))
        serializer = safire.data.serializer.Serializer(pipeline,
                                                       serializer_class,
                                                       **serializer_kwargs)
        pipeline = serializer[pipeline]
    return pipeline


def dry_run(pipeline):
    """Iterates over the pipeline, but doesn't store any results. (

    This is useful just for testing; anytime else, you would be better off just
    serializing whatever you need to iterate through, as it guarantees support
    for various advanced indexing. This only guarantees that all items have
    been processed, initializing various things like document to ID mappings,
    vocabularies...

    :param pipeline: The pipeline over which to iterate.
    """
    for p in pipeline:
        pass


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
            if isinstance(corpus.obj, ShardedCorpus):
                corpus.obj.gensim_retrieval = False
                corpus.obdj.sparse_retrieval = False
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


def smart_cast_dataset(pipeline, **kwargs):
    """Casts the given pipeline to a Dataset with the given kwargs
    unless it already is a Dataset.

    Handles ``test_p`` and ``devel_p`` kwargs by calling pipeline.set_test_p()
    and pipeline.set_devel_p()."""
    if isinstance(pipeline, safire.datasets.dataset.DatasetABC):
        logging.info('Casting pipeline {0} to dataset: already a dataset, '
                     'setting kwargs:'
                     '{1}'.format(pipeline, kwargs))

        if 'test_p' in kwargs:
            pipeline.set_test_p(kwargs['test_p'])
            logging.info('  Set test proportion to {0}, test_doc_offset {1}'
                         ''.format(pipeline.test_p, pipeline._test_doc_offset))

        if 'devel_p' in kwargs:
            pipeline.set_devel_p(kwargs['devel_p'])
            logging.info('  Set devel proportion to {0}, test_doc_offset {1}'
                         ''.format(pipeline.devel_p,
                                   pipeline._devel_doc_offset))

        return pipeline

    else:
        logging.info('Casting pipeline {0} to dataset.'.format(pipeline))
        return safire.datasets.dataset.Dataset(pipeline, **kwargs)


def compute_docname_flatten_mapping(mmdata, mapping_file):
    """Based on a file with docname pairings and document to ID mappings from
    a multimodal dataset, computes the list of indexes that can then be used
    to flatten the given multimodal dataset and work as the text-image mapping
    for pipeline items. (This is different from using the mapping file, because
    multiple data points can share the same document name (sentence vectors
    from a text...).

    Currently works only for a two-component composite dataset.
    """
    t2i_map = safire.utils.parse_textdoc2imdoc_map(mapping_file)
    t2i_list = [[text, image] for text in t2i_map for image in t2i_map[text]]
    t2i_indexes = docnames2indexes(mmdata, t2i_list)
    return t2i_indexes


def mmcorp_from_t_and_i(vtcorp, icorp, ensure_dense=False):
    """Utility function for going from a VTextCorpus (or a pipeline) and
    an ImagenetCorpus (or a pipeline) to a CompositeDataset. Just a shortcut."""
    tdata = smart_cast_dataset(vtcorp, ensure_dense=ensure_dense)
    idata = smart_cast_dataset(icorp, ensure_dense=ensure_dense)
    mmdata = safire.datasets.dataset.CompositeDataset((tdata, idata),
                                                      names=('text', 'img'),
                                                      aligned=False)
    return mmdata


# Not implemented...
def flatten_corpora(corpora, doc_mapping_file):
    """Shortcut function for flattening the given corpora that takes care of
    creating the CompositeDataset and then flattening it again.

    Currently, corpora have to be simple datasets, as some steps during
    flattening do not support recursively composite datasets yet."""
    raise NotImplementedError()


def compute_word2image_map(vtcorp, icorp, t2i_indexes, freqdicts=None):
    """Takes a multimodal dataset with text and images and computes a mapping
    between tokens and images.

    .. note:

        This function should move into some safire.utils.mmutils file, as it
        is very specific to our multimodal setting and not applicable for
        safire pipelines in general.

    :param vtcorp: A VTextCorpus object set to yield documents on iteration.

    :param icorp: An ImagenetCorpus.

    :param t2i_indexes: The mapping between the text and image documents,
        represented as a list of pairs.

    :param freqdicts: A set of frequency dictionaries, one for each document,
        can additionally be supplied. This allows using for example
        tfidf-transformed frequencies instead of the raw counts.

    :return: A pair of dictionaries: w2i, i2w. Tokens are represented in their
        string form (as themselves), images are represented by their image IDs
        (filenames relative to the image root of the data directory).
    """
    # Get documents as their token representations
    documents = [doc for doc in vtcorp.get_texts()]
    word_sets = [frozenset(doc) for doc in documents]
    word_freqs = [freqdict(doc) for doc in documents]

    # Get text name <=> tid mappings
    tname2tid = vtcorp.doc2id  # This will be defaultdict(set) -- ex: sentences
    tid2tname = vtcorp.id2doc  # This will be a dict

    # Get image name <=> iid mappings
    iname2iid = icorp.doc2id
    iid2iname = icorp.id2doc  # This will be a dict

    # Get t2i, i2t mappings
    t2i = collections.defaultdict(list)
    i2t = collections.defaultdict(list)
    for t, i in t2i_indexes:
        t2i[t].append(i)
        i2t[i].append(t)

    # Get word2image mappings
    # For each word, a list of images and how many time was the word with
    # the given image.
    w2i = dict()

    # For each image, a freqdict of tokens
    i2w = dict()

    # Filling in w2i, i2w
    for t in t2i:
        fdict = word_freqs[t]
        iids = t2i[t]
        inames = [iid2iname[iid] for iid in iids]
        #print 'inames for document {0}, tid {1}: {2}'.format(t, tid2tname[t], inames)
        for iname in inames:
            for w in fdict:
                if w not in w2i:
                    w2i[w] = collections.defaultdict(int)
                w2i[w][iname] += fdict[w]
                if iname not in i2w:
                    i2w[iname] = collections.defaultdict(int)
                i2w[iname][w] += fdict[w]

    return w2i, i2w


def docnames2indexes(data, docnames):
    """Converts a mapping of document names to indexes into the given datasets.
    Utility function for flattening datasets that provide a doc2id mapping.

    .. note::

        Currently only supports a non-recursive composite dataset.

    :type data: safire.datasets.dataset.CompositeDataset
    :param data: A composite dataset from which to extract indexing. (This will
        be the dataset you then pass to FlattenDataset.) Currently only works
        with

    :type docnames: list[tuple[str]]
    :param docnames: A list of the document names that should be flattened into
        one item when ``data`` is flattened.

    :rtype: list[tuple[int]]
    :returns: A list of indices into the individual components of the ``data``
        composite dataset.
    """
    doc2ids = [get_doc2id_obj(d) for d in data.data]
    # Problem: returned doc2id object in DocumentFilterCorpus retains the
    # original IDs, not the new ones. We need to convert these IDs
    #print 'Doc2ids:\n  {0}'.format(u'  \n'.join([str(type(d)) for d in doc2ids]))
    #print 'Doc2ids:\n{0}'.format(u'  \n'.join([str(d) for d in doc2ids]))
    output = []
    for name_item in docnames:
        #print 'Name item: {0}'.format(name_item)
        idxs = tuple(doc2ids[i][name] for i, name in enumerate(name_item))
        #print 'Idxs: {0}'.format(idxs)
        # This should work for an empty dict because of defaultdict's behavior.
        output.extend(list(itertools.product(*idxs)))
    #print 'Output: {0}'.format(output)
    return output