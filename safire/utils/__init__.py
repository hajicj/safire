""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
import cProfile
import copy
import logging
import StringIO
import pdb
import pstats
import random
import time

from sys import getsizeof, stderr
from itertools import chain
from collections import deque

#try:
#    from reprlib import repr
#except ImportError:
#    pass


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


try:
    import Image
except ImportError:
    from PIL import Image

import numpy
import numpy.random
import theano
import theano.ifelse

import math


def check_kwargs(kwargs, names):
    """Checks that all \'names\' are in the \'dictionary\'.
    Throws a TypeError if they aren't.

    :type dictionary: dict
    :param dictionary: kwargs dictionary object.

    :type names: iterable
    :param names: argument names that should be in the dictionary.

    :raises: TypeError
    """
    for name in names:
        if name not in kwargs:
            raise TypeError('Missing required kwarg \'%s\'.' % name)


def shared_shape(shared_var, axis):
    """Returns the shape of the shared variable along the given axis.

    :type shared_var: theano.tensor.sharedvar.TensorSharedVariable
    :param shared_var: The shared variable of which we want the shape.

    :type axis: int
    :param axis: Along which axis to compute the shape. 0 means rows,
                 1 means columns.

    :returns: The shape of ``shared_var`` along the given axis ``axis``.
    :rtype: int
    """
    return shared_var.get_value(borrow=True).shape[axis]


def random_shared_var(name, shape, lbound, rbound, rng=numpy.random.RandomState()):
    """Creates a Theano shared variaable with the specified
    shape, with values drawn uniformly from the interval
    ``[lbound, rbound]``.

    :type name: str
    :param name: The name of the shared variable. Has to be supplied.
                 (This is an internal safety measure: if you are creating
                 shared variables, you'd better know what they are!)

    :type shape:
    :param shape: The desired shape of the shared variable.

    :type lbound: theano.config.floatX
    :param lbound: The lower bound of the values with which
                   to initialize the shared variable.

    :type rbound: theano.config.floatX
    :param rbound: The upper bound of the values with which
                   to initialize the shared variable.

    :type rng: numpy.random.RandomState
    :param rng: Optionaly supply a random number generator.
                If ``None`` (default), creates a new one with
                a random seed.

    :rtype: theano.tensor.sharedvar.TensorSharedVariable
    :returns: A shared variable initialized with random values
              drawn uniformly from the interval ``[lbound, rbound]``
              of type theano.config.floatX
    """
    values = numpy.asarray(rng.uniform(low = lbound, high = rbound,
                                       size = shape),
                                       dtype = theano.config.floatX)
    return theano.shared(value = values, name = name)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = numpy.dtype(numpy.uint8) ### ???
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def isqrt(n):
    """Computes the highest integer whose square is less than n.
    Returns this integer and True/False depending on whether this integer
    is actually the square root of n."""
    root = math.sqrt(n)
    iroot = int(root)
    is_equal = (root == float(iroot))
    return iroot, is_equal


##############################################################################

# Profiling

def profile_run(function, *args, **kwargs):
    """Profile an arbitrary function. Returns the results as a StringIO stream
    object."""
    pr = cProfile.Profile()
    pr.enable()

    retval = function(*args, **kwargs)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(.33)

    return s, retval


# Let's try writing a decorator!
def benchmark(fn):
    """Simple timing decorator. Prints to stdout."""
    def _benchmark(*args, **kwargs):
        start = time.clock()
        retval = fn(*args, **kwargs)
        end = time.clock()
        total = end - start
        print 'Benchmark | {0} : {1}'.format(fn.__name__, total)
        return retval
    return _benchmark


def total_size(o, handlers={}):
    """Returns the approximate memory footprint an object and all of its
    contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()                   # track which object id's have already
                                   #  been seen
    default_size = getsizeof(0)    # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        #logging.debug('%s : %s : %s' % (s, type(o), repr(o)))

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


##############################################################################

### Various interesting activation functions

def mask_ReLU(X):
    """Rectified Linear Unit activation function."""
    return (X * (X > 0))


def ReLU(X):
    return theano.tensor.maximum(X, 0.0001)


def cappedReLU(X, cap=2.0):
    return theano.tensor.minimum(theano.tensor.maximum(X, 0.0001), cap)


class build_cappedReLU(object):
    def __init__(self, cap):
        self.cap = cap

    def __call__(self, X):
        return theano.tensor.minimum(theano.tensor.maximum(X, 0.0001), self.cap)


def abstanh(X):
    return theano.tensor.abs_(theano.tensor.tanh(X))

##############################################################################


def detect_nan(i, node, fn):
    """Debugging theano."""
    for output in fn.outputs:
        try:
            if numpy.isnan(output[0]).any():
                print '*** NaN detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [output[0] for output in fn.outputs]
                pdb.set_trace()
                raise ValueError('Found NaN in computation!')
        except TypeError:
            print 'Couldn\'t check node for NaN:'
            theano.printing.debugprint(node)


def merciless_print(i, node, fn):
    """Debugging theano. Prints inputs and outputs at every point."""
    print ''
    print '-------------------------------------------------------'
    print 'Node %s' % str(i)
    theano.printing.debugprint(node)
    print 'Inputs : %s' % [input for input in fn.inputs]
    print 'Outputs: %s' % [output for output in fn.outputs]
    print 'Node:'
    for output in fn.outputs:
        try:
            if numpy.isnan(output[0]).any():
                print '*** NaN detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [output[0] for output in fn.outputs]
                pdb.set_trace()
                raise ValueError('Found NaN in computation!')
            if numpy.isposinf(output[0]).any() or numpy.isneginf(output[0]).any():
                print '*** Inf detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [output[0] for output in fn.outputs]
                pdb.set_trace()
                raise ValueError('Found Inf in computation!')
        except TypeError:
            print 'Couldn\'t check node for NaN/Inf:'
            theano.printing.debugprint(node)


def shuffle_together(*lists):

    zipped_lists = zip(*lists)
    random.shuffle(zipped_lists)
    unzipped_lists = map(list, zip(*zipped_lists))

    print unzipped_lists

    return unzipped_lists


def iter_sample_fast(iterable, samplesize):
    """Got this off stackoverwflow."""
    # I can't remember why it works.
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in xrange(samplesize):
        results.append(iterator.next())
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")
    return results


def uniform_steps(iterable, k):
    """Chooses K items from the iterable so that they are a constant step size
    from each other. The last member is chosen"""
    stepsize = len(iterable) / k
    if stepsize == 0:
        raise ValueError('Too many steps, stepsize 0 (iterable length: %d, steps: %d' % (len(iterable), k))
    output = [ iterable[i-1] for i in range(len(iterable), 0, -stepsize) ]
    output = output[:k]
    return output


def random_steps(iterable, k):
    """Chooses K items uniformly from the iterable."""
    output = random.sample(iterable, k)
    return output


def parse_csv_map(t2i_handle):
    """Given a two-column csv file, returns the forward and backward dicts."""
    t2i = {}
    i2t = {}
    for l in t2i_handle:
        t, i = l.strip().split()
        if t in t2i:
            t2i[t].append(i)
        else:
            t2i[t] = [i]
        if i in i2t:
            i2t[i].append(t)
        else:
            i2t[i] = [t]

    return t2i, i2t

###############################################################################

# Image drawing functions


def compute_column_thumbnail_size(images, total_size, margin=10):
    """Computes the thumbnail size for images so that they all fit into the
    given size. Assumes the images will be arranged underneath each other
    with the given margin around each image (between images, there will be
    two margins)."""

    th_width = int(math.floor(total_size[0] - 2 * margin))
    th_height = int(math.floor((total_size[1] - 2 * len(images) * margin)
                               / len(images)))

    return th_width, th_height


def compute_row_thumbnail_size(images, total_size, margin=10):
    """Computes the thumbnail size for images so that they all fit into the
    given size. Assumes the images will be arranged next to each other
    with the given margin around each image (between images, there will be
    two margins)."""

    th_width = int(math.floor((total_size[0] - 2 * len(images) * margin)
                               / len(images)))
    th_height = int(math.floor(total_size[1] - 2 * margin))

    return th_width, th_height


# Image tiling functions:
def images_to_column(images, margin=10):
    """Tiles images underneath each other."""

    # Compute output image size
    image_sizes = [image.size for image in images]
    max_width = max([isize[0] for isize in image_sizes])

    total_width = max_width + 2 * margin
    total_height = sum([isize[1] for isize in image_sizes]) \
                   + (len(image_sizes) + 1) * margin

    output_image = Image.new('RGB', (total_width, total_height))

    # Compute box positions
    upper_left_corners = [(margin, margin)]
    for image in images[:-1]:
        upper_left_corners.append((margin,
                                   upper_left_corners[-1][1] + image.size[1] + 2 * margin))

    # Paste images to boxes
    for image, upper_left_corner in zip(images, upper_left_corners):
        output_image.paste(image, upper_left_corner)

    return output_image


def images_to_row(images, margin=10):
    """Tiles images next to each other."""

    # Compute output image size
    image_sizes = [image.size for image in images]
    max_height = max([isize[1] for isize in image_sizes])

    total_height = max_height + 2 * margin
    total_width = sum([isize[0] for isize in image_sizes]) \
                   + (len(image_sizes) + 1) * margin

    output_image = Image.new('RGB', (total_width, total_height))

    # Compute box positions
    upper_left_corners = [(margin, margin)]
    for image in images[:-1]:
        upper_left_corners.append((upper_left_corners[-1][0] + image.size[0] + 2 * margin,
                                  margin))

    # Paste images to boxes
    for image, upper_left_corner in zip(images, upper_left_corners):
        output_image.paste(image, upper_left_corner)

    return output_image


def images_to_thumbnails(images, size):
    """Returns a list of image thumbnails of the given size."""
    thumbnails = copy.copy(images)

    for th in thumbnails:
        th.thumbnail(size)

    return thumbnails


def image_comparison_report(images_1, images_2, size=(600, 800), margin=20):
    """Draws the two sets of images as columns side-by-side."""
    colwidth = int(math.floor((size[0] - 4 * margin) / 2))
    colheight = size[1] - 2 * margin

    image_th = compute_column_thumbnail_size(images_1, (colwidth, colheight))

    thumbnails_1 = images_to_thumbnails(images_1, image_th)
    thumbnails_2 = images_to_thumbnails(images_2, image_th)

    cols_1 = images_to_column(thumbnails_1)
    cols_2 = images_to_column(thumbnails_2)

    output = images_to_row([cols_1, cols_2], margin)

    return output


def add_header_image(image, header_image, header_size=(300,200), margin=20):
    """Given an image and a header image, pastes the header image in the
    middle of the input image above the former top of the input image:

    +-----------+                            +----+
    |           |      +----+                |    |
    |           |   +  |    |    ==>         +----+
    +-----------+      +----+             +-----------+
                                          |           |
                                          |           |
                                          +-----------+

    Resizes the header image to the given thumbnail size.
    """

    header_image.thumbnail(header_size)

    h_image_size = header_image.size
    h_strip_size = (image.size[0], h_image_size[1] + 2 * margin)

    output_image = Image.new('RGB', (image.size[0],
                                     image.size[1] + h_strip_size[1]))

    h_image_position = ((h_strip_size[0] - h_image_size[0]) / 2,
                         margin)
    output_image.paste(header_image, h_image_position)
    output_image.paste(image, (0, h_strip_size[1]))

    return output_image

##############################################################################

# Matplotlib plotting


def heatmap_matrix(matrix, title='', with_average=False,
                   colormap='coolwarm', vmin=0.0, vmax=1.0,
                   **kwargs):
    """Creates and shows a heatmap of the given matrix. Optionally, will also
    plot column averages. Doesn't return anything; will show() the figure.

    You can specify heatmap color, bounds and other arguments for
    matplotlib.pyplot.colormesh.

    :type matrix: numpy.ndarray
    :param matrix: The data to plot.

    :type title: str
    :param title: The title of the plotted figure.

    :type with_average: bool
    :param with_average: If set, will plot column averages above the heatmap.

    :type colormap: str
    :param colormap: The colormap to use in colormesh. Passed directly to
        colormesh, so you can use anything you find in the corresponding
        matplotlib doucmentation.

    :type vmin: float
    :param vmin: The value in the ``matrix`` which should correspond to the
        "minimum" color in the heatmap.

    :type vmax: float
    :param vmax: The value in the ``matrix`` which should correspond to the
        "maximum" color in the heatmap.

    :param kwargs: Other arguments to maptlotlib.pyplot.colormesh

    """
    plt.figure(figsize=(matrix.shape[1]*0.002, matrix.shape[0]*0.02),
               dpi=160,
               facecolor='white')
    if with_average:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,2])
        plt.subplot(gs[0])
        plt.title('Average activations')

        # Computing column averages
        avgs = numpy.sum(matrix, axis=0) / matrix.shape[0]

        # Sliding window average
        avg_windowsize = 20
        vscale = 2.0
        wavgs = [ sum(avgs[i:i+avg_windowsize])*vscale / float(avg_windowsize)
                 for i in xrange(matrix.shape[1] - avg_windowsize) ]
        wavgs.extend([avgs[-1] for _ in xrange(avg_windowsize)])

        plt.title(title)
        plt.plot(avgs, 'r,')
        #plt.plot(wavgs, 'b-')

        # Hide x-axis ticks
        frame1 = plt.gca()
        for xlabel_i in frame1.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)

        #plt.xlim([0, matrix.shape[1]])
        plt.subplot(gs[1])

    plt.xlim([0, matrix.shape[1]])
    plt.ylim([0, matrix.shape[0]])
    plt.pcolormesh(matrix, cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)
    if not with_average:
        plt.title(title)
        plt.colorbar()
    plt.show()

##############################################################################


class Noop(object):
    """Doesn't do anything on getitem and call."""
    def __getitem__(self, item):
        return item

    def __call__(self, item):
        return item


##############################

def point_vector(length, point):
    """Creates a numpy ndarray of dimension 1, with the given ``length``, all 0s
    except for position indicated by ``point``, where a 1.0 will be."""
    out = numpy.zeros(length, dtype=theano.config.floatX)


def n_max(arr, n):
    """Find n maximum elements and their indices in a numpy array."""
    indices = arr.ravel().argsort()[-n:]
    indices = (numpy.unravel_index(i, arr.shape) for i in indices)
    return [(arr[i], i) for i in indices]


def squish_array(array):
    """Recomputes array values so that no difference is bigger than 1. Assumes
    the array is sorted."""
    if not array:
        return array
    new_array = [array[0]]
    offset = 0
    for item in array[1:]:
        theoretical_new_item = item - offset
        diff = theoretical_new_item - new_array[-1]
        if diff > 1:
            offset += (diff - 1)
        new_array.append(item - offset)
    return new_array

##################################


def check_malformed_unicode(string):
    """Raises a UnicodeError if the given string cannot be decoded as unicode.
    """
    for letter in string:
        try:
            ul = copy.deepcopy(letter)
            print ul.__repr__()
            ul.encode('raw_unicode_escape').decode('utf-8')
        except:
            print 'Problem encoding as unicode: %s' % letter
            raise


def mock_data_row(dim=1000, prob_nnz=0.5, lam=1.0):
    """Creates a random gensim sparse vector. Each coordinate is nonzero with
    probability ``prob_nnz``, each non-zero coordinate value is drawn from
    a Poisson distribution with parameter lambda equal to ``lam``."""
    nnz = numpy.random.uniform(size=(dim,))
    data = [(i, numpy.random.poisson(lam=lam)) for i in xrange(dim) if nnz[i] < prob_nnz]
    return data


def mock_data(n_items=1000, dim=1000, prob_nnz=0.5, lam=1.0):
    """Creates a mock gensim-style corpus (a list of lists of tuples (int,
    float) to use as a mock corpus.
    """
    data = [mock_data_row(dim=dim, prob_nnz=prob_nnz, lam=lam)
            for _ in xrange(n_items)]
    return data