import collections
import itertools
import unittest

from safire.utils import mock_data
from safire.utils.transcorp import get_id2doc_obj, get_doc2id_obj
from safire.utils.transformers import ItemAggregationTransform, \
    ItemAggregationCorpus
from test.safire_test_case import SafireTestCase, SafireMockCorpus

__author__ = 'Jan Hajic jr'

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


class TestItemAggregation(SafireTestCase):
    def setUp(self):
        self.data = mock_corpus
        self.aggregator = ItemAggregationTransform()
        self.agg_corpus = self.aggregator._apply(self.data)

    def test_initialization_integrity(self):
        self.assertIsInstance(self.agg_corpus, ItemAggregationCorpus)

    def test_iteration(self):
        output_data = [item for item in self.agg_corpus]
        self.assertEqual(len(output_data), n_docs)
        self.assertIs(get_id2doc_obj(self.agg_corpus),
                      self.agg_corpus.id2doc)
        agg_id2doc = get_id2doc_obj(self.agg_corpus)
        self.assertEqual(len(agg_id2doc), n_docs)
        agg_doc2id = get_doc2id_obj(self.agg_corpus)
        for iid in xrange(n_docs):
            self.assertEqual(len(agg_doc2id[iid]), doclengths[iid])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestItemAggregation)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
