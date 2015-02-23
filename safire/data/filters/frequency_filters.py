"""
This module contains classes that ...
"""
import logging

__author__ = "Jan Hajic jr."


def zero_length_filter(document):
    """Filters out documents that are empty.

    :param document: a gensim-style sparse vector

    :return: True if document is not empty, False otherwise.
    """
    return len(document) > 0