"""
This module contains classes that ...
"""
import logging
import itertools
from pprint import pprint, pformat
import safire.utils.transcorp

__author__ = "Jan Hajic jr."


class TfidfBasedTokensFilter(object):
    """This class uses TfIdf information to filter out tokens from a document.
    By default, it expects the normalized TfIdf count and will return tokens
    that round to 1 or more. The exact formula for computing how many tokens
    are retained for a word type is::

      int(10.0 * (f + 0.05))

    The ``0.05`` term can be controlled by the ``freq_add`` init parameter.
    """
    def __init__(self, tfidf_data, freq_add=0.05):
        """

        :param tfidf_data: A pipeline/block that outputs tfidf-transformed
            data on the same corpus on which the filter should be applied,
            so that iids match.

        :param freq_add: Add this much to tfidf output when rounding to nearest
            lower integer.
        """
        self.tfidf_data = tfidf_data
        self.id2word = safire.utils.transcorp.get_id2word_obj(tfidf_data)
        self.word2id = self.id2word.token2id

        self.freq_add = freq_add

        logging.info('Total id2word size: {0}'.format(len(self.id2word)))
        logging.info('Total word2id size: {0}'.format(len(self.word2id)))

    def __call__(self, tokens, doc_iid, sentences=False):
        """Input: list of tokens, doc iid.

        >>> tfidf_data = [[(0, 0.1), (1, 0.23), (4, 0.03), (5, 0.06)]]
        >>> item = [(0, 3), (1, 1), (4, 2), (5, 2)]

        :param sentences: If this flag is set, will consider ``tokens`` not as
            an array of tokens but as an array of arrays of tokens.

        :returns: A list of the tokens that made the cut.
        """
        tfidf = self.tfidf_data[doc_iid]
        tfidf_dict = dict(tfidf)

        # logging.debug(u'Recieved tokens: {0}'.format(pformat(tokens)))

        if sentences:
            all_tokens = itertools.chain(*tokens)
        else:
            all_tokens = tokens
        wids = [self.word2id[token] for token in all_tokens]
        output_freqs = {}
        for wid in wids:
            if wid in tfidf_dict:
                tfidf_based_freq = self.freq_from_tfidf(tfidf_dict[wid])
                output_freqs[self.id2word[wid]] = tfidf_based_freq
            else:
                logging.warn('During tf-idf filtering, a wid of the filtered '
                             'document was not found in the equivalent document'
                             'in the source tf-idf corpus.\n\twid not found: '
                             '{0}\n\tdoc_iid: {1}'.format(wid, doc_iid))
                # skip

        # Generate output (should refactor...)
        if sentences:
            already_passed = {token: 0 for token in output_freqs}
            output_tokens = []
            for sentence in tokens:
                output_sentence = []
                for token in sentence:
                    if token not in output_freqs:
                        continue
                    if already_passed[token] >= output_freqs[token]:
                        continue
                    output_sentence.append(token)
                    already_passed[token] += 1
                output_tokens.append(output_sentence)
        else:
            already_passed = {token: 0 for token in output_freqs}
            output_tokens = []
            for token in all_tokens:
                if token not in output_freqs:
                    continue
                if already_passed[token] >= output_freqs[token]:
                    continue
                output_tokens.append(token)
                already_passed[token] += 1

        return output_tokens

    def freq_from_tfidf(self, tfidf_count):
        return int(10.0 * (tfidf_count + self.freq_add))

