#!/usr/bin/env python
"""
Testing Loader classes:

* safire.data.loaders.MultimodalDatasetLoader

"""

import logging
import os
import unittest

from safire.data.layouts import DataDirLayout
from test import SafireTestCase

class TestLayouts(SafireTestCase):

    def setUp(self):

        self.layout = DataDirLayout('test-data')

        self.test_label = '.test-label'

    def test_required_corpus_names(self):

        reqnames = self.layout.required_corpus_names(self.test_label)
        expected_reqnames = [
            'test-data.test-label.vt.mmcorp',
            'test-data.test-label.vt.mmcorp.index',
            'test-data.test-label.vt.vtcorp',
            'test-data.test-label.img.mmcorp',
            'test-data.test-label.img.mmcorp.index',
            'test-data.test-label.img.icorp'
        ]

        self.assertEqual(expected_reqnames, reqnames)

    def test_required_text_corpus_names(self):

        reqnames = self.layout.required_text_corpus_names(self.test_label)
        expected_reqnames = [
            'test-data.test-label.vt.mmcorp',
            'test-data.test-label.vt.mmcorp.index',
            'test-data.test-label.vt.vtcorp',
        ]

        self.assertEqual(expected_reqnames, reqnames)

    def test_required_img_corpus_names(self):

        reqnames = self.layout.required_img_corpus_names(self.test_label)
        expected_reqnames = [
            'test-data.test-label.img.mmcorp',
            'test-data.test-label.img.mmcorp.index',
            'test-data.test-label.img.icorp'
        ]

        self.assertEqual(expected_reqnames, reqnames)


##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestLayouts)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
