"""
Testing the utility functions.
"""
import gzip

import logging
import math
import os
import unittest

import gensim
from gensim.models import TfidfModel
import operator

from safire.data import VTextCorpus, FrequencyBasedTransformer
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
import safire.utils
from safire.utils.transcorp import bottom_corpus, run_transformations,\
    compute_word2image_map, compute_docname_flatten_mapping
from safire.utils import benchmark

from test.safire_test_case import SafireTestCase


class TestUtils(SafireTestCase):

    @classmethod
    def setUpClass(cls):

        super(TestUtils, cls).setUpClass()

        cls.vtlist_file = os.path.join(cls.data_root, 'test-data.vtlist')
        cls.vtlist = [ os.path.join(cls.data_root, l.strip())
                       for l in open(cls.vtlist_file) ]

        cls.token_filter = PositionalTagTokenFilter(['N', 'A', 'V'], 0)
        cls.strict_token_filter = PositionalTagTokenFilter(['N', 'A'], 0)

    def test_uniform_steps(self):

        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        stepped = safire.utils.uniform_steps(iterable, 4)
        self.assertEqual([10, 8, 6, 4], stepped)

        stepped = safire.utils.uniform_steps(iterable, 10)
        self.assertEqual([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], stepped)

        stepped = safire.utils.uniform_steps(iterable, 6)
        self.assertEqual([10, 9, 8, 7, 6, 5], stepped)

    def test_id2word(self):

        wid = 40

        vtcorp = VTextCorpus(self.vtlist_file, input_root=self.data_root,
                             token_filter=self.token_filter,
                             pfilter=0.3, pfilter_full_freqs=True)
        vtcorp.dry_run()

        vt_word = safire.utils.transcorp.id2word(vtcorp, wid)

        freq_transform = FrequencyBasedTransformer(vtcorp, 110, 10)
        freq_vtcorp = freq_transform[vtcorp]

        freq_vt_word = safire.utils.transcorp.id2word(freq_vtcorp, wid)

        tfidf = TfidfModel(vtcorp)
        tfidf_vtcorp = tfidf[vtcorp]

        tfidf_vt_word = safire.utils.transcorp.id2word(tfidf_vtcorp, wid)

        tfidf_freq = TfidfModel(freq_vtcorp)
        tfidf_freq_vtcorp = tfidf_freq[freq_vtcorp]

        tfidf_freq_vt_word = safire.utils.transcorp.id2word(tfidf_freq_vtcorp, wid)

        freq_tfidf = FrequencyBasedTransformer(tfidf_vtcorp, 110, 10)
        freq_tfidf_vtcorp = freq_tfidf[tfidf_vtcorp]

        freq_tfidf_vt_word = safire.utils.transcorp.id2word(freq_tfidf_vtcorp, wid)

        self.assertEqual(freq_vt_word, tfidf_freq_vt_word)
        self.assertEqual(vt_word, tfidf_vt_word)

        wordlist = [vt_word, freq_vt_word, tfidf_vt_word, tfidf_freq_vt_word,
                    freq_tfidf_vt_word]
        print wordlist

    def test_bottom_corpus(self):

        vtcorp = VTextCorpus(self.vtlist_file, input_root=self.data_root,
                             token_filter=self.token_filter,
                             pfilter=0.3, pfilter_full_freqs=True)
        freq_transform = FrequencyBasedTransformer(vtcorp, 110, 10)
        freq_vtcorp = freq_transform[vtcorp]
        tfidf_freq = TfidfModel(freq_vtcorp)
        tfidf_freq_vtcorp = tfidf_freq[freq_vtcorp]

        self.assertEqual(vtcorp, bottom_corpus(tfidf_freq_vtcorp))
        self.assertEqual(vtcorp, bottom_corpus(freq_vtcorp))

    def test_run_transformations(self):
        vtcorp = VTextCorpus(self.vtlist_file, input_root=self.data_root,
                             token_filter=self.token_filter,
                             pfilter=0.3, pfilter_full_freqs=True)
        vtcorp.dry_run()
        freq_transform = FrequencyBasedTransformer(vtcorp, 110, 10)
        freq_vtcorp = freq_transform[vtcorp]
        tfidf_freq = TfidfModel(freq_vtcorp)

        with gzip.open(self.vtlist[0]) as vt_handle:

            output = run_transformations(vt_handle,
                                         vtcorp,
                                         freq_transform,
                                         tfidf_freq)
        print output
        normalized_output = gensim.matutils.unitvec(output)

        print normalized_output

        self.assertEqual(12, len(output))
        self.assertAlmostEqual(1.0, math.sqrt(sum([f**2 for _, f in output])),
                               delta=0.0001)
        self.assertAlmostEqual(1.0, math.sqrt(sum([f**2 for _, f in normalized_output])),
                               delta=0.0001)

    def test_benchmark(self):

        @benchmark
        def simple_function(a, b):
            result = []
            for i in xrange(a**b):
                result.append([unicode(b) for _ in xrange(b**a)])
            return result

        a = 3
        b = 8
        retval = simple_function(a, b)
        self.assertEqual(len(retval), a**b)
        self.assertEqual(len(retval[0]), b**a)

    def test_compute_word2image_map(self):

        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096, label='')
        icorp.dry_run()

        vtcorp = VTextCorpus(self.vtlist_file, input_root=self.data_root,
                             token_filter=self.strict_token_filter,
                             pfilter=0.3, pfilter_full_freqs=True)
        vtcorp.dry_run()

        idata = Dataset(icorp)
        tdata = Dataset(vtcorp)
        mmdata = CompositeDataset((tdata, idata), names=('text', 'img'),
                                  aligned=False)

        t2i_file = os.path.join(self.loader.root,
                                self.loader.layout.textdoc2imdoc)
        t2i_indexes = compute_docname_flatten_mapping(mmdata, t2i_file)

        w2i, i2w = compute_word2image_map(vtcorp, icorp, t2i_indexes)

        self.assertIsInstance(w2i, dict)
        self.assertIsInstance(i2w, dict)

        print w2i[iter(w2i).next()]
        print w2i['vzduch']
        print w2i['cena-1']
        print i2w['0047938150.jpg']
        print u'\n'.join([u'{0}: {1}'.format(w, freq)
                          for w, freq in sorted(i2w['0047938150.jpg'].items(),
                         key=operator.itemgetter(1), reverse=True)][:10])
        print u'\n'.join([u'{0}: {1}'.format(w, freq)
                          for w, freq in sorted(i2w['0000004532.jpg'].items(),
                         key=operator.itemgetter(1), reverse=True)][:10])





if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestUtils)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)