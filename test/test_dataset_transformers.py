import os
from safire.data import VTextCorpus
from safire.data.sharded_corpus import ShardedCorpus
from safire.datasets.dataset import Dataset, CompositeDataset
from safire.utils import parse_textdoc2imdoc_map
from test.safire_test_case import SafireTestCase
from safire.data.imagenetcorpus import ImagenetCorpus

__author__ = 'Lenovo'

import numpy
import unittest

from safire.datasets.transformations import FlattenComposite, FlattenedDatasetCorpus, \
    docnames2indexes
from safire.data.serializer import Serializer


class TestDatasetTransformers(SafireTestCase):

    def setUp(self):
        # create a composite dataset
        self.full_data = numpy.reshape(numpy.arange(0.0, 100.0, 1.0), (10, 10))
        self.part1 = self.full_data[:, 0:3]
        self.part2 = self.full_data[:, 3:7]
        self.part3 = self.full_data[:, 7:]

        self.data1 = Dataset(self.part1)
        self.data2 = Dataset(self.part2)
        self.data3 = Dataset(self.part3)

        self.composite_partial = CompositeDataset((self.data1, self.data2),
                                                  names=('data1', 'data2'))
        self.composite = CompositeDataset((self.data3, self.composite_partial),
                                          names=('data3', 'composite_partial'))

    def test_flatten_aligned(self):

        flatten = FlattenComposite(self.composite)
        flat = flatten[self.composite]

        flat_batch = flat[0:3]

        self.assertEqual(flat_batch.all(), self.full_data[0:3].all())
        self.assertEqual(flat[3:5].all(), self.full_data[3:5].all())
        self.assertEqual(flat[0:3].all(), self.full_data[0:3].all())

    def test_flatted_indexed(self):

        # Text corpus
        vtlist_filename = os.path.join(self.data_root,
                                       self.loader.layout.vtlist)
        self.vtlist = os.path.join(self.data_root, vtlist_filename)
        self.vtcorp = VTextCorpus(self.vtlist, input_root=self.data_root)
        self.vtcorp.dry_run()
        # (Add some transformations?)

        serialization_tname = 'serialized.vt.shcorp'
        serialization_tfile = os.path.join(self.data_root,
                                           self.loader.layout.corpus_dir,
                                           serialization_tname)
        tserializer = Serializer(self.vtcorp, ShardedCorpus,
                                 serialization_tfile)
        self.vtcorp = tserializer[self.vtcorp]

        # Image corpus
        image_file = os.path.join(self.data_root,
                                  self.loader.layout.image_vectors)
        self.icorp = ImagenetCorpus(image_file, delimiter=';', dim=4096,
                                    label='')
        self.icorp.dry_run()

        serialization_iname = 'serialized.i.shcorp'
        serialization_ifile = os.path.join(self.data_root,
                                           self.loader.layout.corpus_dir,
                                           serialization_iname)
        iserializer = Serializer(self.icorp, ShardedCorpus,
                                 serialization_ifile)
        self.icorp = iserializer[self.icorp]

        # Create multimodal dataset
        tdata = Dataset(self.vtcorp)
        idata = Dataset(self.icorp)

        multimodal = CompositeDataset((tdata, idata), names=('text', 'img'),
                                      aligned=False)

        # Get text-image mapping
        t2i_file = os.path.join(self.data_root,
                                self.loader.layout.textdoc2imdoc)
        with open(t2i_file) as t2i_handle:
            t2i_linecount = sum([1 for _ in t2i_handle])

        t2i_map = parse_textdoc2imdoc_map(t2i_file)
        t2i_list = [[text, image]
                    for text in t2i_map
                    for image in t2i_map[text]]
        t2i_indexes = docnames2indexes(multimodal, t2i_list)

        self.assertIsInstance(t2i_indexes, list)
        self.assertIsInstance(t2i_indexes[0], tuple)
        self.assertIsInstance(t2i_indexes[0][1], int)
        self.assertEqual(len(t2i_indexes), t2i_linecount)

        # Flatten dataset
        flatten = FlattenComposite(multimodal, indexes=t2i_indexes)
        multimodal_flat = flatten[multimodal]

        batch = multimodal_flat[:4]

        self.assertEqual(batch.shape[1], tdata.dim + idata.dim)



###############################################################################


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestDatasetTransformers)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

