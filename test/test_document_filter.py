import unittest
from gensim.utils import SaveLoad
from safire.data.filters.basefilter import BaseFilter
from safire.data.document_filter import DocumentFilterTransform
from safire.utils.transcorp import log_corpus_stack
from test.safire_test_case import SafireTestCase

__author__ = 'Jan Hajic jr'


class OddDocumentFilter(BaseFilter):
    """Filters out documents that have an odd number of items."""
    def passes(self, fields):
        is_even = len(fields) % 2 == 0
        #  print 'Is_even with fields len {0}: {1}'.format(len(fields), is_even)
        return is_even


def odd_document_filter_func(fields):
    """Filters out documents that have an odd number of items."""
    is_even = len(fields) % 2 == 0
    #  print 'Is_even with fields len {0}: {1}'.format(len(fields), is_even)
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
        dfilter = DocumentFilterTransform(OddDocumentFilter())
        docf_corpus = dfilter[self.vtcorp]

        pname = self.loader.pipeline_name('docfiltered')
        docf_corpus.save(pname)
        loaded_corpus = SaveLoad.load(pname)
        print log_corpus_stack(loaded_corpus)
        self.assertIsInstance(loaded_corpus, type(docf_corpus))

        filtered_docs = [d for d in loaded_corpus]
        self.assertEqual(len(filtered_docs), len(self.vtcorp) / 2)

    def test_saveload_func(self):
        dfilter = DocumentFilterTransform(odd_document_filter_func)
        docf_corpus = dfilter[self.vtcorp]

        pname = self.loader.pipeline_name('docfiltered')
        docf_corpus.save(pname)
        loaded_corpus = SaveLoad.load(pname)
        print log_corpus_stack(loaded_corpus)
        self.assertIsInstance(loaded_corpus, type(docf_corpus))

        filtered_docs = [d for d in loaded_corpus]
        self.assertEqual(len(filtered_docs), len(self.vtcorp) / 2)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestDocumentFilter)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

