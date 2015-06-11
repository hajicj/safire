#!/usr/bin/env python
import os

__version__ = '0.0.1'
__author__ = 'Jan Hajic jr.'


def get_test_data_root():
    """Returns the root of the test data. Often used in examples, to simplify
    access to test data in tutorials and such."""
    return os.path.join(os.path.dirname(__file__), '..', 'test', 'test-data')