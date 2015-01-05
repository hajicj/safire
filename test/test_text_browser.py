import os
import operator
from gensim.models import TfidfModel
from safire.data import VTextCorpus, FrequencyBasedTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.data.text_browser import TextBrowser
from test import SafireTestCase

__author__ = 'Lenovo'

import unittest


class TestTextBrowser(SafireTestCase):

    @classmethod
    def setUpClass(cls):
        super(TestTextBrowser, cls).setUpClass()

        cls.vtlist = os.path.join(cls.data_root, 'test-data.vtlist')

        cls.token_filter = PositionalTagTokenFilter(['N'], 0)

        cls.dloader = MultimodalShardedDatasetLoader(cls.data_root, 'test-data')

    def setUp(self):

        self.k = 510
        self.discard_top = 10
        self.vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root,
                                  token_filter=self.token_filter,
                                  pfilter=0.3, pfilter_full_freqs=True)

        self.tfidf = TfidfModel(self.vtcorp)
        self.tfidf_vtcorp = self.tfidf[self.vtcorp]

        self.freq_transform = FrequencyBasedTransformer(self.tfidf_vtcorp,
                                                        self.k,
                                                        self.discard_top)
        self.freq_tfidf_vtcorp = self.freq_transform[self.tfidf_vtcorp]

        self.text_infix = '.NA.top10010.pf0.3.pff.tfidf'
        self.vtcorp.allow_dict_updates = False
        self.browser = TextBrowser(self.data_root, self.freq_tfidf_vtcorp)

    def test_get_text(self):

        text = self.browser.get_text(3)
        lines = text.split('\n')
        #self.assertEqual(291, len(lines))

        text_first5 = self.browser.get_text(2, first_n_sentences=5)
        print text_first5

        first5_lines = text_first5.split('\n')

        self.assertEqual(7, len(first5_lines))

    def test_get_representation(self):

        document = self.browser.get_representation(3)
        text = self.browser.get_text(3, first_n_sentences=10)

        print text

        wdocument = self.browser.get_word_representation(3)
        fwrep = self.browser.format_representation(wdocument)

        print fwrep

        self.assertEqual(len(document), len(wdocument))

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestTextBrowser)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
