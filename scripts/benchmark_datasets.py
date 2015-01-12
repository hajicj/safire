#!/usr/bin/env python
"""
``benchmark_datasets.py`` is a script that compares the running time of several
different implementations of the IndexedCorpus interface.

The measurements will be printed to standard output.

Tasks
-----

For each implementation, we measure performance on the following tasks:

* Serializing a given corpus (``--serialization``, ``-s``),
* In-order iteration through a given corpus (``--inord_iter``),
* In-order iteration in chunks (``--inord_chunks``),
* In-order iteration with conversion to numpy ndarray (``--inord_numpy``),
* In-order iteration with conversion to csr matrices (``--inord_csr``),
* Random access (``--random``),
* Random access with chunks (``--random_chunks``),
* Iteration through the data by chunk, but with a random order of chunks
  (``--shuffle_chunks``)

The option in parentheses will run the given benchmarking task. If you wish
to perform all of them, use ``--all`` (or ``-a``).


Data
----

Given a data root and dataset name (we typically used
``-r $SAFIRE_DATA -n safire_notabloid``), will run a benchmark on the default
text and image datasets. When using smaller datasets, remember that there is
some constant overhead.

Compared implementations
------------------------

* gensim.corpora.MmCorpus
* safire.datasets.sharded_dataset.ShardedDataset

"""
import argparse
import logging
import os
import time
import unittest
from gensim.corpora import MmCorpus
from numpy.random import uniform, poisson, randint
import shutil
from safire.datasets.sharded_dataset import ShardedDataset

from safire.utils import benchmark

__author__ = 'Jan Hajic jr.'

tasks = {
    'serialize': {'name': 'serialization'},
    'inord_iter': {'name': 'in-order iteration'},
    'inord_chunks': {'name': 'in-order iteration by chunk'},
    'inord_numpy': {'name': 'in-order iteration by numpy ndarray'},
    'inord_csr': {'name': 'in-order iteration by scipy csr matrix'},
    'random': {'name': 'random access'},
    'random_chunks': {'name': 'random access by chunk'},
    'shuffle_chunks': {'name': 'iterate by chunk, but in a random order.'}
}


#: How many data points should there be in the mock data
n_items = 10000

#: What the mock data dimension should be
dim = 10000

###############################################################################

# Setting up the show


def mock_data_row(dim=1000, prob_nnz=0.5, lam=1.0):
    """Creates a random gensim sparse vector. Each coordinate is nonzero with
    probability ``prob_nnz``, each non-zero coordinate value is drawn from
    a Poisson distribution with parameter lambda equal to ``lam``."""
    nnz = uniform(size=(dim,))
    data = [(i, poisson(lam=lam)) for i in xrange(dim) if nnz[i] < prob_nnz]
    return data


@benchmark
def mock_data(n_items=1000, dim=1000, prob_nnz=0.5, lam=1.0):
    """Creates a mock gensim-style corpus (a list of lists of tuples (int,
    float) to use as a mock corpus.
    """
    data = [mock_data_row(dim=dim, prob_nnz=prob_nnz, lam=lam)
            for _ in xrange(n_items)]
    return data

###############################################################################

# Running benchmarks


@benchmark
def serialize_mmcorpus(fname, data, **kwargs):
    """Serialization benchmark - MmCorpus"""
    MmCorpus.serialize(fname, data, **kwargs)


@benchmark
def serialize_sharded_dataset(fname, data, **kwargs):
    """Serialization benchmark - ShardedDataset"""
    ShardedDataset.serialize(fname, data, **kwargs)


@benchmark
def iter_inorder(corpus):
    """In-order iteration by individual item."""
    i = 1
    for doc in corpus:
        i += len(doc)


def iter_random(corpus, n_accesses=1000):
    indices = [randint(0, n_accesses-1) for _ in xrange(n_accesses)]
    _iter_random(corpus, indices)


@benchmark
def _iter_random(corpus, indices):
    """Random access by individual item."""
    ctr = 1
    for i, idx in enumerate(indices):
        doc = corpus[idx]
        if i % 500 == 0:
            logging.info('Retrieved doc no. {0}:\n{1}'.format(i, doc))
        ctr += len(doc)

###############################################################################


def main(args):
    logging.info('Executing benchmark_datasets.py...')

    # Create temporary directory for storing all files.
    temp_dir = 'TEMP_benchmarking_safire_datasets'
    if os.path.exists(temp_dir):
        logging.info('Removing the charred skeleton of temp dir {0}'.format(temp_dir))
        shutil.rmtree(temp_dir)
    logging.info('Creating temp dir {0}'.format(temp_dir))
    os.makedirs(temp_dir)

    # Load the data from a mock gensim-style corpus
    logging.info('Creating mock data, {0} items with dimension {1}'.format(
        n_items, dim))
    data = mock_data(dim=args.dim, n_items=args.n_items)

    print '\nSerialization:'
    print '--------------\n'
    print '      shdat...'
    shdat_fname = os.path.join(temp_dir, 'shdat-bench')
    serialize_sharded_dataset(shdat_fname, data, dim=args.dim,
                              shardsize=args.shardsize)

    print '      mmcorpus...'
    mmcorpus_fname = os.path.join(temp_dir, 'mmcorpus-bench')
    serialize_mmcorpus(mmcorpus_fname, data)

    print '\nIn-order retrieval:'
    print '-------------------\n'
    shdat = ShardedDataset.load(shdat_fname)
    print '      shdat-dense2gensim...'
    shdat.gensim = True
    iter_inorder(shdat)
    del shdat

    mmcorpus = MmCorpus(mmcorpus_fname)
    print '      mmcorpus...'
    iter_inorder(mmcorpus)
    del mmcorpus

    print '\nRandom-access iteration:'
    print '------------------------\n'
    indices = [randint(0, args.n_items-1) for _ in xrange(args.n_items)]
    shdat = ShardedDataset.load(shdat_fname)
    print '      shdat-dense2gensim...'
    shdat.gensim = True
    _iter_random(shdat, indices)
    del shdat

    mmcorpus = MmCorpus(mmcorpus_fname)
    print '      mmcorpus...'
    _iter_random(mmcorpus, indices)
    del mmcorpus

    logging.info('Removing temp dir {0}'.format(temp_dir))
    shutil.rmtree(temp_dir)

    logging.info('Exiting benchmark_datasets.py.')


def build_argument_parser():

    parser = argparse.ArgumentParser(description=__doc__, add_help=True)

    # Will not need these arguments, will use mock data???
    # parser.add_argument('-r', '--root', action='store', default=None,
    #                     required=True, help='The path to'+
    #                     ' the directory which is the root of a dataset.' +
    #                     ' (Will be passed to a Loader as a root.)')
    # parser.add_argument('-n', '--name', help='The dataset name passed to the'+
    #                     'Loader. Has to correspond to the *.vtlist file name.')
    parser.add_argument('-n', '--n_items',
                        action='store', type=int, default=1000,
                        help='Use this many items in mock data.')
    parser.add_argument('-d', '--dim', action='store', type=int, default=1000,
                        help='Mock data should have this dimension.')

    # Options for individual tasks.
    for tname, task in tasks.items():
        parser.add_argument('--%s' % task,
                            action='store_true',
                            help='Run the %s task.' % task['name'])

    # Options for ShardedDataset initialization
    parser.add_argument('--shardsize', action='store', default=4096, type=int,
                        help='ShardedDataset corpus shard size.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO logging messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG logging messages. (May get very '
                             'verbose.')

    return parser


if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(format='%(levelname)s : %(message)s',
                            level=logging.INFO)

    main(args)
