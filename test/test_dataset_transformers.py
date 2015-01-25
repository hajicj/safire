from safire.datasets.dataset import Dataset, CompositeDataset
from test.safire_test_case import SafireTestCase

__author__ = 'Lenovo'

import numpy
import unittest

from safire.datasets.transformations import FlattenComposite, FlattenedDataset
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


###############################################################################


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestDatasetTransformers)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)

