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
from safire.utils import freqdict
from test.safire_test_case import SafireTestCase


class TestVTextCorpus(SafireTestCase):

    @classmethod
    def setUpClass(cls):

        super(TestVTextCorpus, cls).setUpClass(clean_only=True)

        cls.vtlist_file = os.path.join(cls.data_root, 'test-data.vtlist')
        with open(cls.vtlist_file) as vtlist_handle:
            cls.vtlist = [line.strip() for line in open(cls.vtlist_file)]

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

    def test_iter_sentences(self):

        self.corpus.sentences = True
        docs = []
        for document in self.corpus:

            #print document
            docs.append(document)

        self.assertEqual(756, len(docs))

        print iter(self.corpus.get_texts()).next()

    def test_iter_tokens(self):

        self.corpus.tokens = True
        docs = []
        for document in self.corpus:

            docs.append(document)

        self.assertEqual(11035, len(docs))

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

    def test_token_min_freq(self):

        self.corpus.token_filter = PositionalTagTokenFilter(['N', 'A'], 0)
        with open(os.path.join(self.corpus.input_root,
                               self.corpus.input)) as vthandle:
            docname = os.path.join(self.corpus.input_root,
                                   vthandle.next().strip())
        with self.corpus._get_doc_handle(docname) as doc_handle:
            doc, sentences = self.corpus.parse_document_and_sentences(doc_handle)

        self.corpus.token_min_freq = 2

        with open(os.path.join(self.corpus.input_root,
                               self.corpus.input)) as vthandle:
            docname = os.path.join(self.corpus.input_root,
                                   vthandle.next().strip())
        with self.corpus._get_doc_handle(docname) as doc_handle:
            doc2, sentences2 = self.corpus.parse_document_and_sentences(doc_handle)

        print '\nFirst doc: {0} tokens'.format(len(doc))
        print u'\n'.join([u'{0}:\t{1}'.format(w, f) for w, f in freqdict(doc).items()])

        print '\nSecond doc: {0} tokens'.format(len(doc2))
        print u'\n'.join([u'{0}:\t{1}'.format(w, f) for w, f in freqdict(doc2).items()])

        self.assertNotEqual(len(doc), len(doc2))

    def test_getitem(self):

        self.corpus.dry_run()

        doc = self.corpus[7]
        with gzip.open(os.path.join(self.data_root, self.vtlist[7])) as vt_handle:
            doc_direct, _ = self.corpus.parse_document_and_sentences(vt_handle)
            doc_direct = self.corpus.doc2bow(doc_direct, allow_update=True)

        self.assertEqual(doc, doc_direct)

        # Caching?
        doc = self.corpus[7]
        self.assertEqual(doc, doc_direct)

    # TODO: tests for doc2id, id2doc

    def test_saveload(self):

        savename = os.path.join(self.data_root,
                                'corpora',
                                'saved.vtcorp')
        self.corpus.save(fname=savename)
        loaded_corpus = VTextCorpus.load(savename)
        print 'Getitem: {0}'.format(self.corpus.__getitem__)
        print 'loaded : {0}'.format(loaded_corpus.__getitem__)
        self.assertIsInstance(loaded_corpus, VTextCorpus)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestVTextCorpus)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)