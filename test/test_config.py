import os
import collections
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.utils.config import ConfigParser, Configuration, ConfigBuilder

__author__ = 'Jan Hajic jr'

import unittest
from test.safire_test_case import SafireTestCase


class TestConfig(SafireTestCase):

    @classmethod
    def setUpClass(cls, clean_only=False, no_datasets=False):
        super(TestConfig, cls).setUpClass(clean_only, no_datasets)
        cls.config_file = os.path.join(cls.data_root, 'test_config.ini')

    def test_parser(self):
        parser = ConfigParser()
        with open(self.config_file) as config_handle:
            conf = parser.parse(config_handle)

        # Correct type returned?
        self.assertIsInstance(conf, Configuration)

        # All objects parsed?
        self.assertIsInstance(conf.objects, collections.OrderedDict)
        self.assertEqual(len(conf.objects), 4)
        for name, obj in conf.objects.items():
            self.assertTrue('_class' in obj)
            if '_dependencies' in obj:
                self.assertIsInstance(obj['_dependencies'], list)
            if '_access_deps' in obj:
                self.assertIsInstance(obj['_access_deps'], dict)
                for dep in obj['_access_deps']:
                    self.assertTrue(dep in obj['_dependencies'])

        # Check for presence of special sections
        self.assertTrue(hasattr(conf, '_info'))
        self.assertTrue(hasattr(conf, '_assembly'))
        self.assertTrue(hasattr(conf, '_builder'))
        self.assertTrue(hasattr(conf, '_loader'))
        self.assertTrue(hasattr(conf, '_persistence'))

    def test_builder(self):
        cparser = ConfigParser()
        with open(self.config_file) as config_handle:
            conf = cparser.parse(config_handle)

        builder = ConfigBuilder(conf)
        self.assertIsInstance(builder, ConfigBuilder)

        # Testing the special sections
        _info = builder._info
        self.assertTrue(hasattr(_info, 'name'))
        self.assertEqual(_info.name, 'test_config')
        _loader = builder._loader
        self.assertTrue(isinstance(_loader, MultimodalShardedDatasetLoader))
        self.assertEqual(_loader.root, 'test-data/')
        self.assertEqual(_loader.layout.name, 'test-data')
        _assembly = builder._assembly
        self.assertTrue(hasattr(_assembly, '_1_'))
        self.assertEqual(_assembly._1_, 'vtcorp')
        self.assertTrue(hasattr(_assembly, '_2_'))
        self.assertEqual(_assembly._2_, 'tfidf[_1_]')
        self.assertTrue(hasattr(_assembly, '_3_'))
        self.assertEqual(_assembly._3_, 'serializer[_2_]')
        _persistence = builder._persistence
        self.assertTrue(hasattr(_persistence, 'loader'))
        for blockname, label in zip(['_1_', '_2_', '_3_'],
                                    ['vtcorp', 'tfidf', 'tfidf.serialized']):
            self.assertTrue(hasattr(_persistence, blockname))
            self.assertEqual(getattr(_persistence, blockname), label)

        # Testing the dependency graph
        deps_graph = builder.deps_graph
        self.assertEqual(len(deps_graph), 7)
        self.assertTrue(deps_graph['tokenfilter'] == set())
        self.assertTrue(deps_graph['vtcorp'] == set(['tokenfilter']))
        self.assertTrue(deps_graph['_1_'] == set(['vtcorp']))
        self.assertTrue(deps_graph['tfidf'] == set(['_1_']))
        self.assertTrue(deps_graph['_2_'] == set(['_1_', 'tfidf']))
        self.assertTrue(deps_graph['serializer'] == set(['_2_', 'tfidf', 'vtcorp']))
        self.assertTrue(deps_graph['_3_'] == set(['_2_', 'serializer']))

        # Testing the dependency graph sort
        sorted_deps = builder.sorted_deps
        expected_sort_order = ['tokenfilter', 'vtcorp', '_1_', 'tfidf', '_2_',
                               'serializer', '_3_']
        for sorted_key, expected_key in zip(sorted_deps, expected_sort_order):
            self.assertEqual(sorted_key, expected_key)

        # Testing persistence resolution
        will_load, will_be_loaded, will_init = builder.resolve_persistence()
        print will_load, will_be_loaded, will_init
        self.assertEqual(len(will_load), 0)
        self.assertEqual(len(will_be_loaded), 0)
        self.assertEqual(len(will_init), 7)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestConfig)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
