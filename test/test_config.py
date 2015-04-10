import os
from safire.utils.config import ConfigParser

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase


class TestConfig(SafireTestCase):

    def setUpClass(cls, clean_only=False, no_datasets=False):
        super(TestConfig, cls).setUpClass(clean_only, no_datasets)
        cls.config_file = os.path.join(cls.data_root, 'test-config.ini')

    def test_parser(self):
        parser = ConfigParser()
        conf = parser.parse(self.config_file)

        # All objects parsed?
        self.assertIsInstance(conf.objects, dict)
        self.assertEqual(len(conf.objects), 5)
        for obj in conf.objects:
            self.assertTrue(hasattr(obj, 'class'))
            self.assertTrue(hasattr(obj, 'init_args'))

        # Check for presence of special sections
        self.assertTrue(hasattr(conf, 'info'))
        self.assertTrue(hasattr(conf, 'assembly'))
        self.assertTrue(hasattr(conf, 'builder'))
        self.assertTrue(hasattr(conf, 'persistence'))


    def test_builder(self):
        self.assertEqual(True, False)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestConfig)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
