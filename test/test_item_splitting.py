import collections
import itertools
import unittest

from safire.utils import mock_data, gensim_batch_nth_member
from safire.utils.transcorp import get_id2doc_obj, get_doc2id_obj
from safire.utils.transformers import SplitDocPerFeatureTransform,\
    SplitDocPerFeatureCorpus
from test.safire_test_case import SafireTestCase, SafireMockCorpus

__author__ = 'Jan Hajic jr'

############################################################################

# Creating mock data
data = [
    [(0, 1.), (1, 2.), (2, 1.), (5, 1.), (6, 2.), (7, 1.)],
    [(0, 2.), (1, 2.), (3, 1.), (4, 1.), (5, 2.), (9, 1.)],
    [(1, 1.), (2, 2.), (4, 1.), (5, 1.), (7, 2.), (8, 1.)],
]
n_items = 3
n_docs = 3
dim = 10
n_nnz = 18
doclengths = [1, 1, 1]
docnames = list(itertools.chain(*[['doc{0}'.format(i)
                                   for _ in xrange(doclengths[i])]
                                  for i in xrange(len(doclengths))]
))
id2doc = collections.defaultdict(str)
for i in xrange(n_items):
    id2doc[i] = docnames[i]

mock_corpus = SafireMockCorpus(data=data, dim=dim, id2doc=id2doc)

############################################################################


class TestItemSplitting(SafireTestCase):

    def setUp(self):
        self.data = mock_corpus
        self.splitter = SplitDocPerFeatureTransform()
        self.split_corpus = self.splitter._apply(self.data)

    def test_initialization_integrity(self):
        self.assertIsInstance(self.split_corpus, SplitDocPerFeatureCorpus)

    def test_iteration(self):
        output_data = [item for item in self.split_corpus]
        self.assertEqual(len(output_data), n_nnz)
        self.assertIs(get_id2doc_obj(self.split_corpus),
                      self.split_corpus.id2doc)
        agg_id2doc = get_id2doc_obj(self.split_corpus)
        self.assertEqual(len(agg_id2doc), n_nnz)
        self.assertEqual(len(self.split_corpus), n_nnz)
        for i in xrange(len(output_data)):
            self.assertEqual(output_data[i], [gensim_batch_nth_member(data, i)])

    def test_getitem(self):
        # Test random-access retrieval through __getitem__:
        # - single int,
        result = self.split_corpus[0]
        self.assertEqual(result, [data[0][0]])  # Format element as gensim vector

        # - slice,
        lower = 4
        upper = 8
        result = self.split_corpus[lower:upper]
        self.assertEqual(len(result), upper - lower)
        # Correctness?
        for i in xrange(lower, upper, 1):
            orig_element = gensim_batch_nth_member(self.data, i)
            self.assertEqual(result[i], [orig_element])

        # - list of indices
        idxs = [0, 5, 2, 3, 11, 5, 0]
        result = self.split_corpus[idxs]
        self.assertEqual(len(result), len(idxs))
        # Correctness?
        for i in xrange(lower, upper, 1):
            orig_element = gensim_batch_nth_member(self.data, i)
            self.assertEqual(result[i], [orig_element])

        pass


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestItemSplitting)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
