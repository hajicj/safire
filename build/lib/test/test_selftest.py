"""
Testing the test procedure itself.
"""

import logging
import unittest


class TestSelfTest(unittest.TestCase):

    def test_test(self):
        result = True
        expected = True
        self.assertEqual(expected, result)

    def test_antitest(self):
        result = False
        expected = True
        self.assertNotEqual(expected, result)

if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()