"""
This module contains classes that ...
"""
import logging
import numpy
import theano
import safire

__author__ = "Jan Hajic jr."


class MultimodalClampedSampler(object):
    """This class implements sampling from a model with a part of
    the inputs clamped to a certain value. This is a utility class
    for performing multimodal experiments.

    It needs a model and the text and image dimensions of the multimodal
    dataset used with the model.
    """

    def __init__(self, model, dim_text, dim_img, heavy_debug=False,
                 starting_img_features=None):
        """Initialize the sampler.

        :type model: safire.learning.models.RestrictedBoltzmannMachine
        :param model: The model from which to sample.

        :type dim_text: int
        :param dim_text: The dimension of the text modality features in the
            model inputs. Counted from feature 0.

        :type dim_img: int
        :param dim_img: The dimension of the image modality features in the
            model inputs. Counted from feature no. ``dim_text``.

        :type heavy_debug: bool
        :param heavy_debug: If set, will compile the sampling functions in
            theano's ``MonitorMode`` with a detailed input/output/node printing
            function.

        :type starting_img_features: numpy.ndarray
        :param starting_img_features: The vector of the initial image features.
            If set to ``None``, will initialize the images to zeros. Must be
            of ``dim_img`` length.
        """
        if not hasattr(model, 'sample_vhv'):
            raise ValueError('Model is not sampleable (class: %s)' % str(
                model.__class__))

        self.model = model

        # Assumes text is always first (default settings for MultimodalDataset,
        # cannot be changed without re-programming the default).
        self.dim_text = dim_text
        self.dim_img = dim_img

        # Common interface...
        self.n_in = dim_text
        self.n_out = dim_img

        if starting_img_features is not None:
            if starting_img_features.shape != (self.dim_img,):
                raise ValueError('Supplied starting image features have shape'
                                 ' {0}, do not match image dimension {1}.'
                                 ''.format(starting_img_features.shape,
                                           self.dim_img))
            self.starting_img_features = starting_img_features
        else:
            self.starting_img_features = numpy.zeros(self.dim_img)

        logging.debug('Sampler dimensions: text %d, images %d' % (self.dim_text,
                                                                  self.dim_img))

        self.heavy_debug = heavy_debug
        self.vhv = self.build_vhv()
        self.vhv_vh_sample = self.build_vhv(sample_hidden=True,
                                            sample_visible=True)
        self.vhv_h_sample = self.build_vhv(sample_hidden=True)
        self.vhv_v_sample = self.build_vhv(sample_visible=True)

    def build_vhv(self, sample_hidden=False, sample_visible=False):
        """Creates and compiles the vhv mean sampling function."""
        v = theano.tensor.matrix('vhv_mean_in', dtype=theano.config.floatX)
        v.tag.test_value = numpy.ones((10, self.dim_img + self.dim_text),
                                      dtype=theano.config.floatX)

        if sample_hidden:
            h = self.model.sample_h_given_v(v)
        else:
            h = self.model.mean_h_given_v(v)

        if sample_visible:
            v_prime = self.model.sample_v_given_h(h)
        else:
            v_prime = self.model.mean_v_given_h(h)

        vhv_kwargs = {}
        if self.heavy_debug:
            vhv_kwargs['mode'] = theano.compile.MonitorMode(
                post_func=safire.utils.merciless_print).excluding(
                    'local_elemwise_fusion', 'inplace')

        vhv = theano.function(inputs=[v],
                              outputs=v_prime,
                              allow_input_downcast=True,
                              **vhv_kwargs)
        return vhv

    def t2i_step(self, text_features, image_init_features=None,
                 sample_hidden=False, sample_visible=False):
        """Runs the theano compiled function that implements clamped sampling
        with fixed text and sampled images once.

        Assumes text features come first in model input.

        :param sample: If given, will sample hidden and visible layer instead
            of computing mean.
        """
        # Should be churning out matrices, not individual vectors.
        numpy.atleast_2d(text_features)

        # Input shape has to have the same number of rows as the input.
        inputs = numpy.zeros((len(text_features), self.model.n_in),
                             dtype=theano.config.floatX)
        #logging.info('Initialized inputs shape: {0}, text_features shape: {1}'
        #             ''.format(inputs.shape, text_features.shape))
        inputs[:, :self.dim_text] = text_features

        # Assumes the image init features have the right shape.
        if image_init_features is not None:
            inputs[:, self.dim_text:] = image_init_features

        # Magic happens here (gibbs vhv step)
        #print 'Setting gibbs vhv step: sample_hidden {0}, sample_visible {1}' \
        #      ''.format(sample_hidden, sample_visible)
        if sample_hidden and sample_visible:
            outputs = self.vhv_vh_sample(inputs)
        elif sample_hidden:
            outputs = self.vhv_h_sample(inputs)
        elif sample_visible:
            outputs = self.vhv_v_sample(inputs)
        else:
            outputs = self.vhv(inputs)

        image_features = outputs[:, self.dim_text:]
        return image_features

    def t2i_run_chain(self, text_features, k=1,
                      sample_hidden=False,
                      sample_visible=False):
        """Runs the chain for K steps. Set whether to sample hidden
        and/or visible layer."""

        img_features = self.get_img_init_features(text_features)
        for step in xrange(k):
            # Use the new image features and the old text features.
            img_features = self.t2i_step(text_features, img_features,
                                         sample_hidden=sample_hidden,
                                         sample_visible=sample_visible)

        return img_features

    def t2i_run_chain_mean_last(self, text_features, k=1,
                                sample_hidden=True, sample_visible=True):
        """Runs the chain for K steps. In all steps except last, both layers
        are sampled. In the last step, both the hidden and visible layer use
        means. If k=1, then, no sampling is used at all."""
        img_features = self.t2i_run_chain(text_features, k-1,
                                          sample_hidden=sample_hidden,
                                          sample_visible=sample_visible)
        img_features = self.t2i_step(text_features, img_features,
                                     sample_hidden=False,
                                     sample_visible=False)
        return img_features

    def get_img_init_features(self, text_features):
        """Returns the default image init features."""
        image_init_features = numpy.zeros((len(text_features), self.dim_img)) \
                              + self.starting_img_features
        return image_init_features
