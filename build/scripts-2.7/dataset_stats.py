#!c:\users\lenovo\canopy\user\scripts\python.exe
"""
``dataset_stats.py`` is a script that collects statistics about datasets. It
computes:

* Mean, variance

* Number of zero elements, nonzero elements, sparsity

* Histogram of feature values, in steps of 0.01 (or specified by argument)

* Histogram of average feature values, in steps of 0.01

* Visualization of feature averages

* Average feature correlation? Weighed by feature sums?

* Direct visualization of a part of the dataset

"""
import argparse
import logging
import cPickle
import random
import gensim
from gensim.utils import SaveLoad
import numpy
import matplotlib.pyplot as plt
import sys
import theano
import time
import safire
from safire.utils.matutils import scale_to_unit_covariance, generate_grid, \
    grid2sym_matrix
from safire.data.loaders import MultimodalShardedDatasetLoader
from safire.utils.transcorp import dimension, convert_to_dense

__author__ = 'Jan Hajic jr.'

class DatasetStats(object):
    """A simple container for various statistics about a dataset."""

    ## @profile
    # def _slice_dataset(self, dataset, indexes):
    #     """Workaround because of some weird memory leak."""
    #     loading_interval = 20
    #     d = None
    #     for group in gensim.utils.grouper(indexes, loading_interval):
    #         dg = numpy.array([dataset[idx] for idx in group])
    #         if d is None:
    #             d = dg
    #         else:
    #             d = numpy.vstack((d, dg))
    #     return d

    def _nnz_cell_tracking(self, cell, item_nnz):
        # Non-zero cells tracking
        if cell != 0.0:
            item_nnz += 1
        return item_nnz

    def _feature_totals_update(self, cell, f):
        self.feature_totals[f] += cell

    def _nnz_item_tracking(self, item_nnz):
        self.num_nnz += item_nnz
        nnz_hist_key = item_nnz - (item_nnz % 10)
        self.nnz_histogram_bins[nnz_hist_key] += 1

    def _hist_cells_from_item(self, cell):
        # Cell histogram update
        hist_key = int(100 * numpy.around(cell, 2))
        # Generate 0's if current cell highest in histogram
        if hist_key > self.max_hist_bin:
            logging.info('Increasing max hist key from %f to %f' % (
                self.max_hist_bin, hist_key))
            for hk in numpy.arange(self.max_hist_bin, hist_key + 99, 1):
                self.histogram_bins[int(hk)] = 0.0
            self.max_hist_bin = hist_key
        self.histogram_bins[hist_key] += 1.0

    def _init_process_item(self, feature_totals, item, item_nnz):
        for f, cell in enumerate(item):
            self._feature_totals_update(cell, f)

            item_nnz = self._nnz_cell_tracking(cell, item_nnz)

            # if cell > maximum:
            #    maximum = cell
            #if cell < minimum:
            #    minimum = cell

            #self._hist_cells_from_item(cell)

        # self.maximum = max(self.maximum, maximum)
        #self.minimum = min(self.minimum, minimum)
        self._nnz_item_tracking(item_nnz)

    def __init__(self, dataset, sample_n=None):
        """Collects statistics on a dataset. Tries to do so efficiently, i.e. all in
        one pass.

        Computes:

        * Average per cell (when the dataset is viewed as a matrix, a cell is the
          value of a feature in an item)

        * Average per cell per feature

        :type dataset: safire.data.sharded_dataset.ShardedDataset
        :param dataset: A dataset to compute statistics on.

        :type sample_n: int
        :param sample_n: Instead of running through the whole corpus, only
            run through a sample of this many items.
        """
        self.d_idxs = []
        if sample_n:
            self.d_idxs = sorted([ numpy.random.choice(range(len(dataset)),
                                                       replace=False)
                                   for _ in xrange(sample_n) ])
            logging.info('Completed sampling.')
            logging.debug('Dataset samples:')
            for idx in self.d_idxs[:10]:
                logging.debug(dataset[idx])
        else:
            self.d_idxs = range(len(dataset))

        self.dataset = dataset

        logging.info('Dataset total: %d' % len(dataset))

        start = int(time.clock())

        #print 'self.d_idxs: {0}/{1}'.format(type(self.d_idxs),
        #                                    type(self.d_idxs[0]))
        #print 'self.dataset: {0}/{1}'.format(type(self.dataset),
        #                                     type(self.dataset[self.d_idxs[0]]))
        #print 'data: {0}'.format(self.dataset[self.d_idxs[0]])
        self.item_totals = [sum(self.dataset[idx]) for idx in self.d_idxs]
        self.dataset_total = sum(self.item_totals)

        current_time = time.clock()
        logging.info('Dataset total complete in %d s' % (current_time - start))

        self.dataset_dim = len(self.dataset[0])
        self.n_items_processed = len(self.d_idxs)
        self.n_cells_processed = len(self.d_idxs) * self.dataset_dim
        self.average_cell = self.dataset_total / self.n_cells_processed

        self.num_nnz = 0.0

        self.maximum = max([max(self.dataset[idx]) for idx in self.d_idxs])
        self.minimum = min([min(self.dataset[idx]) for idx in self.d_idxs])

        current_time = time.clock()
        logging.info('Maxima and minima complete in %d s' % (current_time - start))

        # For each non-zero cell value range of 0.01: how many cells in this range?
        self.histogram_bins = { 0 : 0.0 }
        self.max_hist_bin = 0

        # For each integer range of 10: how many items with this many nnz cells?
        self.nnz_histogram_bins = {int(i) : 0.0
                                   for i in numpy.arange(0, self.dataset_dim + 9, 10)}


        # The sum of cells in each column
        self.feature_totals = numpy.zeros(self.dataset_dim,
                                          dtype=theano.config.floatX)

        iterating_start_time = time.clock()

        for n_th, idx in enumerate(self.d_idxs):
            item = numpy.array(self.dataset[idx])
            self.feature_totals += item
            item_nnz = 0.0
            #self._init_process_item(self.feature_totals, item, item_nnz)

            #
            # if (n_th + 1) % 50 == 0:
            #     logging.info('Processed %d items in %d s' % (n_th+1,
            #     time.clock() - iterating_start_time))

        #self.percentage_nnz = self.num_nnz / self.n_cells_processed

    def report(self):
        """Prints out the stats, with appropriate formatting."""
        print 'Dataset:'
        print '\tLength:\t%f' % self.n_items_processed
        print '\tDimension:\t%d' % len(self.dataset[0])
        print '\tAverage:\t%.5f' % self.average_cell
        print '\tMinimum:\t%f' % self.minimum
        print '\tMaximum:\t%f' % self.maximum
        #print '\tNonzero:\t%f' % self.num_nnz
        #print '\tPerc. Nonzero:\t%.3f' % self.percentage_nnz

    def cell_histogram(self):
        """Plots the cell value density."""
        # Discard the 0 key, since it would dominate too much
        keys = sorted(self.histogram_bins.keys())[1:]
        values = [ self.histogram_bins[hk] for hk in keys ]
        keys = [ hk / 100.0 for hk in keys ] # Scale back
        plt.plot(keys, values)
        plt.show()

    def raw_cell_histogram(self, n=100, n_bins=100):
        """Plots the cell value histogram directly from the dataset,
        taking the first N items."""
        data = self.dataset[0:n]
        data = numpy.ravel(data, order='K') # Faster?
        data = [ d for d in data if d != 0.0 ] # Remove zeros

        plt.hist(data, n_bins, histtype='step', normed=0)
        plt.show()

    def cell_mass_histogram(self, n=10, n_bins=100):
        """Plots the mass in each histogram cell."""
        data = [ self.dataset[i] for i in self.d_idxs[0:(max([n, self.n_items_processed]))] ]
        data = numpy.ravel(data, order='K')
        data = [ d for d in data if d != 0.0 ]
        plt.figure()
        n, bins, patches = plt.hist(data, n_bins, histtype='step', normed=0)
        mass = numpy.array([ nn * b for nn, b in zip(n, bins)])

        plt.plot(bins[:-1], mass, color='r')
        plt.show()

    def cell_cumulative_mass(self, n=100, n_bins=100):
        data = [ self.dataset[i] for i in self.d_idxs[0:(max([n, self.n_items_processed]))] ]
        data = numpy.ravel(data, order='K')
        data = [ d for d in data if d != 0.0 ]
        plt.figure()
        n, bins, patches = plt.hist(data, n_bins, histtype='step', normed=0)
        plt.clf() # Histogram shouldn't show up
        mass = numpy.array([ nn * b for nn, b in zip(n, bins)])

        cumulative_mass = numpy.cumsum(mass)
        # Normalize
        cumulative_mass = cumulative_mass / cumulative_mass[-1]

        # Find 0.9 and 0.95 quantiles
        mass90 = None
        mass95 = None
        for i, c in enumerate(cumulative_mass):
            if c > 0.9 * cumulative_mass[-1] and mass90 is None:
                mass90 = i
            if c > 0.95 * cumulative_mass[-1] and mass95 is None:
                mass95 = i

        #fn = lambda p: (p + p**2) / (p + p**2 + 1.7)
        fn = lambda p: p**1.4 / (p**1.4+1.0)
        approx = [ fn(x) for x in bins[:-1]]

        plt.xlim([0,bins[mass95]])
        plt.ylim([0,1])
        plt.plot(bins[:-1], cumulative_mass, color='g')
        plt.plot(bins[:-1], approx, color='b')

        plt.scatter([bins[mass90]], [cumulative_mass[mass90]], c='g')
        plt.scatter([bins[mass95]], [cumulative_mass[mass95]], c='g')
        plt.vlines([bins[mass90], bins[mass95]],
                   [0,0],
                   [cumulative_mass[mass90], cumulative_mass[mass95]],
                   linestyles='dashed')
        plt.show()


    def nnz_histogram(self):
        """Plots the distribution of nonzero counts."""
        # Discard the 0 key, since it would dominate too much
        keys = sorted(self.nnz_histogram_bins.keys())[1:]
        values = [ self.nnz_histogram_bins[k] for k in keys ]
        plt.plot(keys, values)
        plt.show()

    def plot_feature(self, feature):
        """Plots the counts for a feature."""
        values = [ self.dataset[i][feature] for i in self.d_idxs ]
        sorted_values = sorted(values)
        plt.figure()
        plt.plot(values)
        plt.plot(sorted_values, color='r')
        plt.show()

    def rich_histogram(self, values, n_bins=100, with_zero=False, title=None):
        values = values.ravel(order='K') # For multi-dimensional data
        nonzero_values = [v for v in values if v != 0.0]
        num_zero = len(values) - len(nonzero_values)
        p_zero = float(num_zero) / float(len(values))

        if with_zero:
            nonzero_values = values

        plt.figure()
        n, bins, patches = plt.hist(nonzero_values,
                                    bins=n_bins, histtype='step', color='b')
        mass = numpy.array([nn * b for nn, b in zip(n, bins)])
        nz_mean = numpy.average(nonzero_values)
        mean = numpy.average(values)
        n_saturated = n[-1]
        p_saturated = n_saturated / float(len(values))

        saturation_threshold = bins[-2]
        saturated = numpy.array([ v for v in nonzero_values
                                  if v > saturation_threshold ])
        saturation_avg = numpy.average(saturated)

        plt.ylim([0, max(n)*1.1])

        plt.vlines([mean], 0, max(n) / 2, linestyles='dashed', colors='r')
        plt.text(mean, max(n) / 2 + 0.4, str(mean), color='r')

        plt.vlines([nz_mean], 0, max(n) / 4, linestyles='dashed', colors='g')
        plt.text(nz_mean, max(n) / 4 + 0.4, str(nz_mean), color='g')

        plt.text(mean, max(n) / 1.2, 'P(0) = %.5f' % p_zero)
        plt.text(mean, max(n) / 1.3, 'P(sat.) = %.5f, avg(sat.) = %.5f' % (
            p_saturated, saturation_avg))

        plt.plot(bins[:-1], mass, 'r')

        if title:
            plt.title(title)
        # plt.annotate(str(mean), xy=(mean, max(n) / 2), xytext=(mean, max(n)/2 + 0.4),
        # arrowprops=dict(facecolor='white'), color='r')
        plt.show()

    def feature_histogram(self, feature, n_bins=None, whole_dataset=False):
        """Plots the histogram of feature values and the mean. Does not include
        zero values, only gives their proportion."""
        if whole_dataset:
            idxs = range(len(self.dataset))
        else:
            idxs = self.d_idxs

        values = numpy.array([self.dataset[i][feature] for i in idxs])
        if n_bins is None:
            n_bins = len(values) / 10
        self.rich_histogram(values, n_bins=n_bins, title='Feature %d' % feature)

    def item_histogram(self, idx, n_bins=None):
        if n_bins is None:
            n_bins = self.dataset.dim / 33
        self.rich_histogram(self.dataset[idx], n_bins=n_bins, title='Item %d' % idx)

    def feature_grid(self, features, fn):
        """
        Evaluates given function ``fn`` average over all pairs of given features.

        :param features:
        :param fn:
        :return:
        """
        values = numpy.array([ self.dataset[i][features] for i in self.d_idxs ])
        # If there are two features: return their MSE
        #if (len(features) == 2):
        #    return fn(values[:, 0], values[:, 1])
        #else:
        return generate_grid(values, fn)

    #def plot_feature_mses(self, features):

    def feature_lecunn_covariances(self, n_items=1000):
        """Computes and plots feature covariances, according to Y. LeCunn's
        1998 paper on efficient backprop.

        Uses the first ``n_items`` data points."""
        covariances = numpy.sum(self.dataset[0:n_items] ** 2, axis=1) / n_items
        plt.figure()
        plt.plot(covariances)
        plt.show()

    def pairwise_feature_matrix(self, features, fn):
        pairwise_grid = self.feature_grid(features, fn)
        pairwise_matrix = grid2sym_matrix(pairwise_grid)
        return pairwise_matrix

###########################################################################


def do_stats_init(args, dataset):
    stats = DatasetStats(dataset, sample_n=args.sample)  # Takes LONG.
    return stats


def main(args):
    logging.info('Executing dataset_stats.py...')

    loader = MultimodalShardedDatasetLoader(args.root, args.name)

    # Loading and/or computing
    if not args.load:
        dataset_name = loader.pipeline_name(args.dataset)
        dataset = SaveLoad.load(dataset_name)
        dataset = convert_to_dense(dataset)
        # if args.text:
        #     wrapper_dataset_name = loader.pipeline_name(args.dataset)
        #     wrapper_dataset = SaveLoad.load(wrapper_dataset_name)
        #     dataset = wrapper_dataset.data
        #     vtcorp = wrapper_dataset.vtcorp
        #     print 'Dimension of underlying text data: %d' % dimension(vtcorp)
        #     print 'Dimension of dataset: %d' % dataset.dim
        #     # The ShardedDataset, not the text-modality wrapper
        # else:
        #     dataset = loader.load_img(args.dataset).data

        logging.info('Loaded dataset: %d items, dimension %d' % (len(dataset), dimension(dataset)))
        report, stats = safire.utils.profile_run(do_stats_init, args, dataset)
    else:
        with open(args.load) as input_handle:
            stats = cPickle.load(input_handle)

    stats.report()
    #stats.raw_cell_histogram()
    #stats.cell_mass_histogram(n=100)
    #stats.cell_cumulative_mass(n=100)

    max_feature = list(stats.feature_totals).index(max(stats.feature_totals))
    #stats.feature_histogram(0, stats.n_items_processed, whole_dataset=True)
    #stats.feature_histogram(max_feature,
    #                       stats.n_items_processed, whole_dataset=True)

    #stats.item_histogram(0)
    #stats.item_histogram(3)
    #stats.item_histogram(117)
    #stats.feature_lecunn_covariances()

    #stats.nnz_histogram()


    inspected_features = sorted([ numpy.random.choice(stats.dataset_dim,
                                                      replace=False)
                                  for _ in range(100)])
    #inspection_matrix = stats.pairwise_feature_matrix(inspected_features,
    #                                       safire.utils.matutils.maxn_sparse_rmse)
    #safire.utils.heatmap_matrix(inspection_matrix, 'MaxNormalized dataset RMSE')
    #pairwise_avg = numpy.average(inspection_matrix)

    logging.info('Sampling raw matrix...')

    n_raw_samples = min(len(dataset), 1000)
    raw_matrix = numpy.array([stats.dataset[idx]
                              for idx in stats.d_idxs[:n_raw_samples]])

    if args.activation_histogram:
        logging.info('Computing histogram of activations...')
        stats.rich_histogram(raw_matrix.ravel(), n_bins=100,
                             with_zero=False,
                             title='Feature activation histogram')

    if args.average_activations:
        logging.info('Computing average activations...')
        feature_totals = numpy.array(stats.feature_totals)
        avg_feature_totals = feature_totals / numpy.sum(feature_totals)
        plt.plot(sorted(avg_feature_totals))
        plt.hist(avg_feature_totals, bins=20, color='red', histtype='step',
                 orientation='horizontal')
        plt.title('Sorted feature means')
        plt.show()

    if args.normalize_covariance:
        logging.info('Normalizing covariance...')
        covariances = numpy.sum(raw_matrix ** 2, axis=0) / raw_matrix.shape[0]
        #print covariances[:10]
        scaled_raw_matrix = scale_to_unit_covariance(raw_matrix)
        scaled_covariances = numpy.sum(scaled_raw_matrix ** 2, axis=0) / scaled_raw_matrix.shape[0]

        plt.figure()
        plt.plot(covariances, color='b')
        plt.plot(scaled_covariances, color='r')
        plt.show()

        #stats.feature_histogram(max_feature, n_bins=100, whole_dataset=True)
        #stats.rich_histogram(raw_matrix[:,max_feature], n_bins=100)
        #stats.rich_histogram(scaled_raw_matrix[:,max_feature], n_bins=100)

        #stats.rich_histogram(raw_matrix[:,0], n_bins=100)
        #stats.rich_histogram(scaled_raw_matrix[:,0], n_bins=100)

        safire.utils.heatmap_matrix(numpy.absolute(scaled_raw_matrix),
                                    title='UCov. dataset heatmap',
                                    with_average=True,
                                    colormap='afmhot',
                                    vmin=0.0, vmax=stats.maximum)


    if args.raw:
        safire.utils.heatmap_matrix(numpy.absolute(raw_matrix),
                                    title='Dataset heatmap',
                                    with_average=True,
                                    colormap='afmhot',
                                    vmin=0.0, vmax=stats.maximum)

        stats.rich_histogram(raw_matrix.ravel())

    if args.correlation:
        logging.info('Computing correlation...')
        if args.normalize_covariance:
            corrcoef_matrix = numpy.corrcoef(scaled_raw_matrix, rowvar=0)
        else:
            corrcoef_matrix = numpy.corrcoef(raw_matrix, rowvar=0)

        print 'Average correlation: %f' % numpy.average(corrcoef_matrix)

        plt.figure(facecolor='white', figsize=(8,6))
        plt.pcolormesh(numpy.absolute(corrcoef_matrix),
                       #title='Pearson C.Coef. heatmap',
                       cmap='afmhot',
                       vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xlim([0,corrcoef_matrix.shape[1]])
        plt.ylim([0,corrcoef_matrix.shape[0]])
        plt.show()

    if args.tanh:
        logging.info('Plotting tanh transformation...')
        tanh_matrix = numpy.tanh(raw_matrix / args.tanh)
        stats.rich_histogram(tanh_matrix, n_bins=20, title='Tanh matrix histogram.')

    if args.hyper:
        logging.info('Plotting hyperbolic transformation...')
        hyp_matrix = raw_matrix / (raw_matrix + args.hyper)
        stats.rich_histogram(hyp_matrix, n_bins=100, title='x/(1+x) matrix histogram.')

    if args.sparsity:
        logging.info('Computing sparsity...')
        # One entry per feature, counts how many non-zero elements are there
        # in each column of the raw matrix.
        num_nnz  = numpy.array([ len([i for i in raw_matrix[:,f] if f != 0])
                                  for f in range(raw_matrix.shape[1]) ],
                               dtype=numpy.float32)
        p_nnz = num_nnz / float(raw_matrix.shape[0])
        plt.plot(sorted(p_nnz))
        plt.hist(p_nnz, bins=20, histtype='stepped', color='r',
                 orientation='horizontal')


    #print 'Pairwise average for %d random features: %f' % (len(inspected_features),
    #                                                       pairwise_avg)

    if args.save:
        with open(args.save, 'w') as output_handle:
            cPickle.dump(stats, output_handle, protocol=-1)

    logging.info('Exiting dataset_stats.py.')


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-r', '--root', required=True,
                        help='The root dataset directory, passed to Loader.')
    parser.add_argument('-n', '--name', required=True,
                        help='The name passed to Loader.')

    parser.add_argument('-d', '--dataset', default='',
                        help='The dataset infix.')
    parser.add_argument('--text', action='store_true',
                        help='Is the dataset a text dataset? If not set, '
                             'assumes image dataset.')

    parser.add_argument('--sample', default=None, type=int,
                        help='If the dataset is large, we may just take a '
                             'sample of this size from it.')

    parser.add_argument('--average_activations', action='store_true',
                        help='Plot sorted average activations of neurons '
                             '& histogram.')
    parser.add_argument('--activation_histogram', action='store_true',
                        help='Plot a rich histogram of activations.')
    parser.add_argument('--raw', action='store_true',
                        help='Plot a heatmap of the raw dataset.')
    parser.add_argument('--correlation', action='store_true',
                        help='Plot a heatmap of feature correlation.')
    parser.add_argument('--normalize_covariance', action='store_true',
                        help='If true, normalizes sample LeCunn covariance and'
                             ' plots a the 0th and max-mean '
                             'feature hitograms before and after scaling..')
    parser.add_argument('--tanh', action='store', type=float, default=None,
                        help='Squish sample through tanh, plot histogram.')
    parser.add_argument('--hyper', action='store', type=float, default=None,
                        help='Squish sample through x/(1+x), plot histogram.')
    parser.add_argument('--sparsity', action='store_true',
                        help='Computes average number of non-zero elements per'
                             ' feature.')

    parser.add_argument('-s', '--save',
                        help='Pickle the stats object to this file.')
    parser.add_argument('-l', '--load',
                        help='Unpickle the stats object from this file instead'
                             ' of computing the stats from the dataset.')

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
