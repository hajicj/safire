"""
This module contains classes that ...
"""
import logging
import safire.utils.transcorp

__author__ = "Jan Hajic jr."


class TfidfBasedTokensFilter(object):
    """This class uses TfIdf information to filter out tokens from a document.
    By default, it expects the normalized TfIdf count and will return tokens
    that round to 1 or more. The exact formula for computing how many tokens
    are retained for a word type is::

      int(10.0 * (f + 0.05))
    """
    def __init__(self, tfidf_data):
        """

        :param tfidf_data: A pipeline/block that outputs tfidf-transformed
            data on the same corpus on which the filter should be applied,
            so that iids match.
        """
        self.tfidf_data = tfidf_data
        self.id2word = safire.utils.transcorp.get_id2word_obj(tfidf_data)
        self.word2id = self.id2word.token2id

    def __call__(self, tokens, doc_iid):
        """Input: list of tokens, doc iid.

        :returns: A list of the tokens that made the cut.
        """
        tfidf = self.tfidf_data[doc_iid]
        tfidf_dict = dict(tfidf)
        wids = [self.word2id[token] for token in tokens]
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
        already_passed = {token: 0 for token in output_freqs}
        output_tokens = []
        for token in tokens:
            if token not in output_freqs:
                continue
            if already_passed[token] >= output_freqs[token]:
                continue
            output_tokens.append(token)
            already_passed[token] += 1

        return output_tokens

    def freq_from_tfidf(self, tfidf_count):
        return int(10.0 * (tfidf_count + 0.05))

