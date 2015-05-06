"""
This module contains utility functions for looking at what's going on
inside the pipeline, but in a way that doesn't fall inside the ``introspection``
module way.

Essentially, all stuff in here should be later refactored into
``introspection``. The functions here are shortcuts.
"""
import logging
import operator
import safire.utils.transcorp as transcorp
import safire.utils.matutils as matutils

__author__ = "Jan Hajic jr."

###############################################################################


def make_token_iid2word_fn(plain_token_corpus):
    """Creates a function that will return the token belonging to a given
    item in a corpus where tokens are documents. This enables us to link
    non-BOW representations of tokens, if there is one token per document, to
    the tokens themselves for an analysis of how the representations behave.

    The function will need:

    * A way of retrieving which token is represented by the i-th document,
    * a translation of the BOW representation of the token to the original.

    The id2word mapping can be retrieved from the plain token corpus.

    :return: A function ``iid2word`` that will return the token corresponding
        to the iid-th document in the token corpus.
    """
    id2word = transcorp.get_id2word_obj(plain_token_corpus)

    def iid2word(iid):
        item = plain_token_corpus[iid]
        wid = item[0][0]
        token = id2word[wid]
        return token

    return iid2word


def most_similar_items(corpus, query, limit=10000, k=10,
                       similarity=matutils.cosine_similarity,
                       selfsim=False, iid_map_fn=None):
    """Finds the ``k`` items most similar to ``item`` in the first ``limit``
     items of ``corpus``. The distance used is ``similarity``.

     If ``selfsim`` is ``False``, will discard all items with similarity
     to query greater than 0.9999. If ``iid_map_fn`` is given, will
     not use iids as similarity keys, will use iids run through the function
     instead. (Useful for mapping iids corresponding to same documents/items
     to one.)"""
    similarities = {}
    for iid, x in enumerate(corpus):
        if iid >= limit:
            break
        sim = similarity(query, x)
        if sim >= 0.9999 and not selfsim:
            continue
        else:
            if iid_map_fn is not None:
                simkey = iid_map_fn(iid)
            else:
                simkey = iid
            similarities[simkey] = sim

    sorted_similarities = sorted(similarities.items(),
                                 key=operator.itemgetter(1))
    # From most to least similar
    return reversed(sorted_similarities[-k:])