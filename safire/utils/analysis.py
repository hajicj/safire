"""
This module contains utility functions for looking at what's going on
inside the pipeline, but in a way that doesn't fall inside the ``introspection``
module way.

Essentially, all stuff in here should be later refactored into
``introspection``. The functions here are shortcuts.
"""
import codecs
import collections
import logging
import operator
import os
import math
import safire.introspection.html_utils as html_utils
import safire.utils.transcorp as transcorp
import safire.utils.matutils as matutils

__author__ = "Jan Hajic jr."

###############################################################################


class HtmlInteractiveWriter(object):
    """The InteractiveWriter is intendend for situations when you are exploring
    a finished pipeline and want to visualize whatever is on your mind.
    The writer only provides an "open window" to a temporary html file and
    expects you to provide the HTML code corresponding to things you want to
    see. It will then provide the necessary boilerplate and paste your html
    content there.

    It doesn't inherit from the WriterABC because it does *not* involve any
    corpus. It can stand perfectly separately.
    """
    def __init__(self, fname, imgs_root=None, id2word=None):
        """The root is quite superficial here."""
        self.fname = os.path.abspath(fname)
        self.imgs_root = os.path.abspath(imgs_root)
        self.id2word = id2word

    def write(self, value):
        header = self.get_html_header()
        body = self.get_html_body(value)
        footer = self.get_html_footer()

        output = header + u'\n' + body + u'\n\n' + footer
        with codecs.open(self.fname, 'w', 'utf-8') as output_handle:
            output_handle.write(output + u'\n')

    def doc_as_img(self, doc):
        """Utility function for speeding up writing images."""
        if self.imgs_root is not None:
            img_fname = os.path.join(self.imgs_root, doc)
        else:
            img_fname = doc
        img_url = html_utils.as_local_url(img_fname)
        value = html_utils.as_image(img_url, width=300, height=180)
        captioned_value = html_utils.as_table([[value], [doc]])
        return captioned_value

    def docs_as_imgs(self, docs):
        """Utility function to write multiple images from their docnames."""
        imgs = [self.doc_as_img(d) for d in docs]
        row_length = int(math.sqrt(len(imgs) + 1))

        n_dummy_imgs = row_length - (len(imgs) % row_length)
        imgs.extend(['' for _ in xrange(n_dummy_imgs)])

        tabulated_imgs = [imgs[start:start+row_length]
                          for start in range(0,
                                             len(imgs) - row_length,
                                             row_length)]
        value = html_utils.as_table(tabulated_imgs)
        return value

    def wids_to_vocabulary(self, wids):
        """Utility function to write a list of wids as a vocabulary grid."""
        tokens = [self.id2word[wid] for wid in wids]
        return u'\n'.join(tokens)

    def get_html_content(self, value):
        """``value`` is already expected to be html code. Future interactive
        writers may redefine this."""
        return value

    def get_html_body(self, value):
        """Formats the content as the ``<body>`` element of an html file."""
        content = self.get_html_content(value)
        return html_utils.text_with_tag(content, 'body', newline=True)

    @staticmethod
    def get_html_header():
        """Generates the constant part of the HTML output that comes *before*
        the content."""
        return unicode(html_utils.head)

    @staticmethod
    def get_html_footer():
        """Generates the constant part of the HTML output that comes *after*
        the content."""
        return u'\n'.join(
            [html_utils.as_comment(u'Generated using HtmlInteractiveWriter.'),
             unicode(html_utils.foot)])


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


def get_token_word2iid_obj(iid2word, plain_token_corpus):
    """Creates a dict that for a given word returns the set of all iids at which
    the word occurs.

    :param iid2word: A function that maps from iids of token documents to the
        words that the item with the given iid represents.

    :param plain_token_corpus: The corpus from which the iids should be aggregated
        by word they represent.

    :return: A defaultdict with words as keys and sets of iids as values.
    """
    word2iid = collections.defaultdict(list)
    for iid in xrange(len(plain_token_corpus)):
        word = iid2word(iid)
        word2iid[word].append(iid)
    return word2iid


def make_token_word2iid_fn(iid2word, plain_token_corpus):
    word2iid_obj = get_token_word2iid_obj(iid2word, plain_token_corpus)

    def word2iid(word):
        return word2iid_obj[word]

    return word2iid


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