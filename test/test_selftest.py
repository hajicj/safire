"""
Testing the test procedure itself.
"""

import logging
import unittest
from test import SafireTestCase


class TestSelfTest(SafireTestCase):

    def test_test(self):
        result = True
        expected = True
        self.assertEqual(expected, result)

    def test_antitest(self):
        result = False
        expected = True
        self.assertNotEqual(expected, result)

##############################################################################

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TestSelfTest)
    suite.addTest(tests)
    runner = unittest.TextTestRunner()
    runner.run(suite)
