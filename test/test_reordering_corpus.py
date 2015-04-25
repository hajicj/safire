import itertools
import collections
from safire.utils import mock_data
from safire.utils.transformers import ReorderingTransform, ReorderingCorpus

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase, SafireMockCorpus

############################################################################

# Creating mock data
dim = 1000
n_items = 100
data = mock_data(n_items=n_items, dim=dim, prob_nnz=0.3)
n_docs = 7
doclengths = [10, 20, 10, 1, 19, 30, 10]
docnames = list(itertools.chain(*[['doc{0}'.format(i)
                                   for _ in xrange(doclengths[i])]
                                  for i in xrange(len(doclengths))]
))
id2doc = collections.defaultdict(str)
for i in xrange(n_items):
    id2doc[i] = docnames[i]

mock_corpus = SafireMockCorpus(data=data, dim=dim, id2doc=id2doc)

############################################################################


class TestReorderingCorpus(SafireTestCase):
    def setUp(self):
        self.mapping = [0, 0, 0, 1, 1, 1, 9, 9, 9, 3, 3, 3, 0, 0, 1, 99]
        self.transformer = ReorderingTransform(self.mapping)
        self.corpus = self.transformer[mock_corpus]

    def tearDown(self):
        del self.mapping
        del self.transformer
        del self.corpus

    def test_init(self):
        self.assertIsInstance(self.transformer, ReorderingTransform)
        self.assertIsInstance(self.corpus, ReorderingCorpus)

        # Test proper id2doc/doc2id initialization
        self.assertEqual(len(self.corpus.id2doc), len(self.mapping))
        expected_docnames = set(mock_corpus.id2doc[i] for i in self.mapping)
        self.assertEqual(len(self.corpus.doc2id),
                         len(expected_docnames))
        for i, x in enumerate(self.mapping):
            self.assertEqual(self.corpus.id2doc[i], mock_corpus.id2doc[x])
            for j, y in enumerate(self.mapping):
                if x == y:
                    self.assertEqual(self.corpus.id2doc[x],
                                     self.corpus.id2doc[y])
                    self.assertEqual(self.corpus.doc2id[i],
                                     self.corpus.doc2id[j])

    def test_len(self):
        self.assertEqual(len(self.corpus), len(self.mapping))

    def test_iter(self):
        items = [x for x in self.corpus]
        for i, item in enumerate(items):
            self.assertEqual(item, data[self.mapping[i]])

    def test_getitem(self):
        for i in xrange(len(self.corpus)):
            item = self.corpus[i]
            self.assertEqual(item, data[self.mapping[i]])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestReorderingCorpus)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
