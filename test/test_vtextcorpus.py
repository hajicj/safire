#!/usr/bin/env python
"""
Testing Loader classes:

* safire.data.loaders.MultimodalDatasetLoader

"""

import gzip
import logging
import os
import unittest
import itertools
import operator
from safire.data.loaders import MultimodalShardedDatasetLoader

from safire.data.vtextcorpus import VTextCorpus
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from test import SafireTestCase

class TestVTextCorpus(SafireTestCase):

    @classmethod
    def setUpClass(cls):

        super(TestVTextCorpus, cls).setUpClass(clean_only=True)

        cls.vtlist_file = os.path.join(cls.data_root, 'test-data.vtlist')
        with open(cls.vtlist_file) as vtlist_handle:
            cls.vtlist = [ line.strip() for line in open(cls.vtlist_file)]

        cls.token_filter = PositionalTagTokenFilter(['N', 'A', 'D', 'V'], 0)

    def setUp(self):

        self.corpus = VTextCorpus(self.vtlist_file, input_root=self.data_root)
        self.filtered_corpus = VTextCorpus(self.vtlist_file,
                                          input_root=self.data_root,
                                          token_filter=self.token_filter)
        self.pfiltered_corpus = VTextCorpus(self.vtlist_file,
                                            input_root=self.data_root,
                                            pfilter=0.3,
                                            pfilter_full_freqs=True)

    def tearDown(self):

        del self.corpus
        del self.filtered_corpus

    def test_parse_sentences(self):

        with gzip.open(os.path.join(self.data_root, self.vtlist[0])) as document_handle:

            sentences = self.corpus.parse_sentences(document_handle)
            self.assertEqual(25, len(sentences))
            self.assertEqual(10, len(sentences[-1]))

        # Testing with filtering. Specific numbers according to
        # test-data/text/idn-00000.vt.txt.gz
        with gzip.open(os.path.join(self.data_root, self.vtlist[0])) as document_handle:

            f_sentences = self.filtered_corpus.parse_sentences(document_handle)

            self.assertEqual(25, len(f_sentences))
            self.assertEqual(7, len(f_sentences[-1]))

    def test_iter(self):

        docs = []
        for document in self.corpus:

            #print document
            docs.append(document)

        self.assertEqual(10, len(docs))

    def test_pfilter(self):

        docs = []
        pdocs = []
        for doc, pdoc in itertools.izip(self.corpus, self.pfiltered_corpus):

            doc_words = [ self.corpus.dictionary[w]
                          for w in itertools.imap(operator.itemgetter(0), doc)]
            freqs = [ f for f in itertools.imap(operator.itemgetter(1), doc)]
            doc_with_words = zip(doc_words, freqs)

            pdoc_words = [ self.pfiltered_corpus.dictionary[w]
                          for w in itertools.imap(operator.itemgetter(0), pdoc)]
            pfreqs = [ f for f in itertools.imap(operator.itemgetter(1), pdoc)]
            pdoc_with_words = zip(pdoc_words, pfreqs)

            # All word in PDoc are in doc
            set_dw = frozenset(doc_words)
            for pw in pdoc_words:
                self.assertTrue(pw in set_dw)

            # Frequencies should match, pfiltered_corpus uses pfilter_full_freqs
            ddict = dict(doc)
            pdict = dict(pdoc)
            for pw in pdoc_words:
                d_freq = ddict[self.corpus.dictionary.token2id[pw]]
                p_freq = pdict[self.pfiltered_corpus.dictionary.token2id[pw]]
                self.assertEqual(d_freq, p_freq)

            docs.append(doc)
            pdocs.append(pdoc)

        self.assertTrue(len(docs) == len(pdocs))

    # What is this test???
    def test_tfidf(self):

        tfidf_corpus = VTextCorpus(self.vtlist_file,
                                   input_root=self.data_root,
                                   pfilter=0.3,
                                   pfilter_full_freqs=True)

        with gzip.open(os.path.join(self.data_root, self.vtlist[0])) as document_handle:
            tdoc, tsentences = tfidf_corpus.parse_document_and_sentences(
                document_handle)
            tbow = tfidf_corpus.doc2bow(tdoc)
        with gzip.open(os.path.join(self.data_root, self.vtlist[0])) as document_handle:
            doc, sentences = self.pfiltered_corpus.parse_document_and_sentences(
                document_handle)
            bow = self.pfiltered_corpus.doc2bow(doc)

        print tbow
        print bow

        tcorp_infix = '.pf0.3.pff.tfidf'
        dloader = MultimodalShardedDatasetLoader(self.data_root, 'test-data')
        dloader.save_text_corpus(tfidf_corpus, tcorp_infix)

        loaded_tcorp = dloader.load_text_corpus(tcorp_infix)
        with gzip.open(os.path.join(self.data_root, self.vtlist[0])) as document_handle:
            doc, sentences = loaded_tcorp.parse_document_and_sentences(
                document_handle)
            ltbow = loaded_tcorp.doc2bow(doc)

        print ltbow

        self.assertEqual(tbow, ltbow)

        self.assertEqual(len(tsentences), len(sentences))

    def test_getitem(self):

        self.corpus.dry_run()

        try:
            with gzip.open(os.path.join(self.data_root, self.vtlist[0])) as vt_handle:

                output = self.corpus[vt_handle]

            print output
        except Exception:
            self.assertEqual(True, False)
            raise
        else:
            self.assertTrue(True)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestVTextCorpus)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)