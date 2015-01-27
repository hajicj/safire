#!/usr/bin/env python
"""
``multimodal_pipeline.py`` is a script that trains a multimodal pipeline
from some test data. It serves as a tutorial on how to set up a complex training
scenario with multiple data sources.
"""
import argparse
import logging
from safire.data.filters.positionaltagfilter import PositionalTagTokenFilter

__author__ = 'Jan Hajic jr.'

vtcorp_settings = {'token_filter': PositionalTagTokenFilter(['N', 'A', 'V'], 0),
                   'pfilter': 0.2,
                   'pfilter_full_freqs': True,
                   'filter_capital': True,
                   'precompute_vtlist': True}
vtlist_fname = 'test-data.vtlist'

freqfilter_settings = {'k': 110,
                       'discard_top': 10}

tanh = 0.5

serialization_vtname = 'serialized.vt.shcorp'
serialization_iname = 'serialized.i.shcorp'



if __name__ == '__main__':

    pass
