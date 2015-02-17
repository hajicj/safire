import unittest
from safire.data.filters.basefilter import BaseFilter
from safire.data.document_filter import DocumentFilterTransform
from test.safire_test_case import SafireTestCase

__author__ = 'Jan Hajic jr'


class OddDocumentFilter(BaseFilter):
    """Filters out documents that have an odd number of items."""
    def passes(self, fields):
        is_even = len(fields) % 2 == 0
        print 'Is_even with fields len {0}: {1}'.format(len(fields), is_even)
        return is_even


class TestDocumentFilter(SafireTestCase):

    def test_apply(self):
        dfilter = DocumentFilterTransform(OddDocumentFilter())
        docf_corpus = dfilter[self.vtcorp]

        filtered_docs = [doc for doc in docf_corpus]
        print 'Total filtered docs: {0} left out of {1}' \
              ''.format(len(filtered_docs), len(self.vtcorp))
        self.assertEqual(len(filtered_docs), len(self.vtcorp) / 2)

    def test_saveload_obj(self):
        self.assertEqual(True, False)

    def test_saveload_func(self):
        self.assertEqual(True, False)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestDocumentFilter)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

