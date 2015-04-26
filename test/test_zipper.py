import itertools
import collections
import numpy
import os
from safire.data.composite_corpus import CompositeCorpus, Zipper
from safire.datasets.transformations import FlattenComposite
from safire.utils import mock_data, gensim2ndarray
from safire.utils.transcorp import compute_docname_flatten_mapping, dimension
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


class TestZipper(SafireTestCase):

    def setUp(self):
        self.corpus_1 = mock_corpus
        self.corpus_2 = mock_corpus

        self.zipper = Zipper((self.corpus_1, self.corpus_2))
        self.zipper_f = Zipper((self.corpus_1, self.corpus_2), flatten=True)

    def tearDown(self):
        del self.corpus_1
        del self.corpus_2
        del self.zipper
        del self.zipper_f

    def test_init(self):
        self.assertIsInstance(self.zipper, Zipper)

        # Dimension derived correctly
        self.assertEqual(self.zipper.dim, (dim, dim))
        self.assertEqual(self.zipper_f.dim, 2 * dim)

    def test_apply(self):
        composite = self.zipper._apply((self.corpus_1, self.corpus_2))
        self.assertIsInstance(composite, CompositeCorpus)
        self.assertEqual(dimension(composite), self.zipper.dim)

    def test_getitem(self):
        item = (self.corpus_1[0], self.corpus_2[0])
        self.assertEqual(item, self.zipper[item])

    def test_getitem_flattened_numpy(self):
        item = (gensim2ndarray(self.corpus_1[0], dimension(self.corpus_1)),
                gensim2ndarray(self.corpus_2[0], dimension(self.corpus_2)))
        self.assertEqual(Zipper.flatten_numpy(item).all(),
                         self.zipper_f[item].all())

    def test_getitem_flattened_gensim(self):
        item = (self.corpus_1[0], self.corpus_2[0])
        self.assertEqual(Zipper.flatten_gensim(item, self.zipper.dim),
                         self.zipper_f[item])

    def test_flatten_gensim(self):
        item_1 = [(0, 1), (1, 3), (4, 5)]
        item_2 = [(0, 2), (2, 1), (4, 1)]
        d = (7, 5)

        expected = [(0, 1), (1, 3), (4, 5), (7, 2), (9, 1), (11, 1)]
        self.assertEqual(expected, Zipper.flatten_gensim((item_1, item_2), d))

    def test_flatten_numpy(self):
        item_1 = numpy.array([1, 3, 0, 0, 5, 0, 0])
        item_2 = numpy.array([2, 0, 1, 0, 1])
        expected = numpy.array([1, 3, 0, 0, 5, 0, 0, 2, 0, 1, 0, 1])
        self.assertEqual(expected.all(), Zipper.flatten_numpy((item_1, item_2)).all())

    def test_zipper_and_composite(self):
        # Tests flattening texts and images through reordering.
        zipper_mm = Zipper((self.vtcorp_serialized, self.icorp_serialized),
                           names=('txt', 'img'), flatten=False)
        mm_data = zipper_mm._apply((self.vtcorp_serialized,
                                    self.icorp_serialized))
        t2i_indexes = compute_docname_flatten_mapping(
            mm_data,
            os.path.join(self.loader.root, self.loader.layout.textdoc2imdoc))

        t_mapping, i_mapping = zip(*t2i_indexes)
        t_reorder = ReorderingTransform(t_mapping)
        vtcorp_reordered = t_reorder[self.vtcorp_serialized]
        i_reorder = ReorderingTransform(i_mapping)
        icorp_reordered = i_reorder[self.icorp_serialized]

        zipper = Zipper((vtcorp_reordered, icorp_reordered),
                        names=('txt', 'img'))
        composite = zipper._apply((vtcorp_reordered, icorp_reordered))
        single_item = composite[0]
        sliced_item = composite[:4]
        list_item = composite[[0, 4, 2, 6]]

        zipper_f = Zipper((vtcorp_reordered, icorp_reordered),
                          names=('txt', 'img'), flatten=True)
        composite_f = zipper_f._apply((vtcorp_reordered, icorp_reordered))
        single_item_f = composite_f[0]
        sliced_item_f = composite_f[:4]
        list_item_f = composite_f[[0, 4, 2, 6]]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestZipper)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
