import os
from PIL import Image
import webbrowser
from safire.data.serializer import Serializer
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
from safire.datasets.transformations import FlattenComposite
from safire.introspection.interfaces import IntrospectionTransformer
from safire.introspection.writers import HtmlSimpleWriter, HtmlStructuredFlattenedWriter
from safire.utils.transcorp import get_id2word_obj, \
    compute_docname_flatten_mapping, log_corpus_stack

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase


class TestIntrospection(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        super(TestIntrospection, cls).setUpClass(clean_only, no_datasets)

        cls.icorp.dry_run()
        cls.image_file = os.path.join(cls.data_root,
                                      cls.loader.layout.img_dir,
                                      cls.icorp.id2doc.values()[0])

    def test_can_load_image(self):

        image = Image.open(self.image_file)
        image.show()

    # def test_print_text_and_image(self):
    #
    #     image = Image.open(self.image_file)
    #
    #     id2word_obj = get_id2word_obj(self.vtcorp)
    #     doc = iter(self.vtcorp).next()
    #     words = [u'{0}'.format(id2word_obj[d[0]]) for d in doc]
    #
    #     print u'\n'.join([w for w in words])

    def test_html_introspection(self):

        transformer = IntrospectionTransformer(
            self.vtcorp,
            root=os.path.join(self.data_root,
                              self.loader.layout.introspection_dir))
        introspected_vtcorp = transformer[self.vtcorp]
        filenames = [fname_vect for fname_vect in introspected_vtcorp]
        # Note that each fname is a list, as the introspection transformer
        # outputs *vectors of length 0*
        print 'Filenames: {0}'.format(filenames)

        for fname_vect in filenames:
            self.assertTrue(os.path.isfile(fname_vect[0]))
        webbrowser.open(filenames[0][0])

    def test_composite_introspection(self):

        tserializer = Serializer(self.vtcorp, ShardedCorpus,
                                 self.loader.pipeline_serialization_target('.text'),
                                 gensim_serialization=False, gensim_retrieval=False)
        vtcorp = tserializer[self.vtcorp]
        tdata = Dataset(vtcorp, ensure_dense=True)

        iserializer = Serializer(self.icorp, ShardedCorpus,
                                 self.loader.pipeline_serialization_target('.img'),
                                 gensim_serialization=False, gensim_retrieval=False)
        icorp = iserializer[self.icorp]
        idata = Dataset(icorp, ensure_dense=True)

        mmdata = CompositeDataset((tdata, idata), names=('txt', 'img'),
                                  aligned=False)

        t2i_file = os.path.join(self.loader.root,
                                self.loader.layout.textdoc2imdoc)
        t2i_indexes = compute_docname_flatten_mapping(mmdata, t2i_file)

        flatten = FlattenComposite(mmdata, indexes=t2i_indexes, structured=True)
        mm_pipeline = flatten[mmdata]
        # mm_results = [mm_vect for mm_vect in mm_pipeline]

        twriter = HtmlSimpleWriter(root=self.loader.root)
        iwriter = HtmlSimpleWriter(root=self.loader.root)
        composite_writer = HtmlStructuredFlattenedWriter(root=self.loader.root,
                                               writers=(twriter, iwriter))

        introspection = IntrospectionTransformer(mm_pipeline,
                                                 writer=composite_writer)
        introspected_pipeline = introspection[mm_pipeline]
        filenames = [fname_vect for fname_vect in introspected_pipeline]
        for fname_vect in filenames:
            self.assertTrue(os.path.isfile(fname_vect[0]))
        webbrowser.open(filenames[0][0])

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestIntrospection)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
