import os
from PIL import Image
import webbrowser
from safire.introspection.interfaces import IntrospectionTransformer
from safire.utils.transcorp import get_id2word_obj

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



if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestIntrospection)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
