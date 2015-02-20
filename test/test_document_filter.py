import unittest
from gensim.utils import SaveLoad
import os
from safire.data.filters.basefilter import BaseFilter
from safire.data.document_filter import DocumentFilterTransform
from safire.data.imagenetcorpus import ImagenetCorpus
from safire.datasets.transformations import FlattenComposite
from safire.utils.transcorp import log_corpus_stack, mmcorp_from_t_and_i, \
    compute_docname_flatten_mapping
from test.safire_test_case import SafireTestCase

__author__ = 'Jan Hajic jr'


class OddDocumentFilter(BaseFilter):
    """Filters out documents that have an odd number of items."""
    def passes(self, fields):
        is_even = len(fields) % 2 == 0
        #  print 'Is_even with fields len {0}: {1}'.format(len(fields), is_even)
        return is_even


class LargeDocumentFilter(BaseFilter):
    """Filters out documents where any feature is larger than 4.0."""
    def passes(self, fields):
        for i, f in enumerate(fields):
            if f[1] > 40.0:
                print 'Field {0} larger than limit: {1} > {2}'.format(i, f, 4.0)
                return False
        return True


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

    def test_flatten_dryrun(self):
        icorp = ImagenetCorpus(os.path.join(self.loader.root,
                                            self.loader.layout.image_vectors),
                               delimiter=';', dim=4096)
        icorp.dry_run()
        self.vtcorp.dry_run()

        ifilter = DocumentFilterTransform(LargeDocumentFilter())
        icorp_filtered = ifilter[icorp]

        idocs = [d for d in icorp_filtered]
        print 'Images after filtering: {0}'.format(len(idocs))

        tfilter = DocumentFilterTransform(OddDocumentFilter())
        tcorp_filtered = tfilter[self.vtcorp]

        tdocs = [d for d in tcorp_filtered]
        print 'Texts after filtering: {0}'.format(len(tdocs))

        t2i_file = os.path.join(self.loader.root,
                                self.loader.layout.textdoc2imdoc)
        with open(t2i_file) as t2i_handle:
            lines = [l.strip() for l in t2i_handle]
            print 'Total lines in t2i file: {0}'.format(len(lines))

        mmdata = mmcorp_from_t_and_i(tcorp_filtered, icorp_filtered)
        t2i_indexes = compute_docname_flatten_mapping(mmdata, t2i_file)

        print 'Total t2i indexes: {0}'.format(len(t2i_indexes))

        flatten = FlattenComposite(mmdata, t2i_indexes)
        mmcorp = flatten[mmdata]

        docs = [d for d in mmcorp]
        print 'Total docs after flattening: {0}'.format(len(docs))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestDocumentFilter)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

