"""
.. module:: 
    :platform: Unix
    :synopsis: ???

.. moduleauthor: Jan Hajic <hajicj@gmail.com>
"""
import cPickle
import numpy
import safire.utils

import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

from safire.learning.models.autoencoder import Autoencoder

class DenoisingAutoencoder(Autoencoder):
    """This is a dummy docstring for class . You had better write a real one.
    
    """
    def __init__(self, inputs, n_in, n_out=100, 
                 activation=TT.nnet.sigmoid,
                 backward_activation=TT.nnet.sigmoid,
                 reconstruction='cross-entropy',
                 W=None, W_prime=None, b=None, b_prime=None, 
                 tied_weights=True,
                 corruption_level=0.3,
                 L1_norm=0.0, L2_norm=0.0, bias_decay=0.0,
                 sparsity_target=None, output_sparsity_target=None,
                 rng=numpy.random.RandomState(), 
                 theano_rng=None):
        """ Initialize the parameters of the Denoising Autoencoder.
        A Denoising Autoencoder is an unsupervised model that tries to minimize
        reconstruction error on input with additional noise introduced to the
        model.

        The noise randomly switches off input neurons with a certain
        probability. This is different from a *dropout training* procedure,
        where the *hidden* neurons are randomly switched off.

        :type inputs: theano.tensor.TensorType
        :param inputs: Symbolic variable that descripbes the input
                       of the architecture (e.g., one minibatch of
                       input images, or output of a previous layer)


        :type n_in: int
        :param n_in: Number of input units, the dimension of the space
                     in which the data points live

        :type n_out: int
        :param n_out: The number of hidden units.
        
        :type activation: theano.tensor.elemwise.Elemwise
        :param activation: The nonlinearity applied at neuron
                           output.
                      
        :type W: theano.tensor.sharedvar.TensorSharedVariable
        :param W: Theano variable pointing to a set of weights that should
                  be shared between the autoencoder and another architecture;
                  if autoencoder should be standalone, leave this as None.
                  This set of weights refers to the transition from visible
                  to hidden layer.
        
        :type W_prime: theano.tensor.sharedvar.TensorSharedVariable
        :param W_prime: Theano variable pointing to a set of weights that
                        should be shared between the autoencoder and another
                        architecture; if autoencoder should be standalone,
                        leave this as None. This set of weights refers to
                        the transition from the hidden to the visible layer.
                        
        :type b: theano.tensor.sharedvar.TensorSharedVariable
        :param b: Theano variable pointing to a set of bias values that
                  should be shared between the autoencoder and another
                  architecture; if autoencoder should be standalone,
                  leave this as None. This set of bias values refers
                  to the transition from visible to hidden layer. 
                  
        :type b_prime: theano.tensor.sharedvar.TensorSharedVariable
        :param b_prime: Theano variable pointing to a set of bias values
                        that should be shared between the autoencoder and
                        another architecture; if autoencoder should be 
                        standalone, leave this as None. This set of bias
                        values refers to the transition from visible to 
                        hidden layer. 
                        
        :type tied_weights: bool
        :param tied_weights: If True (default), forces W_prime = W.T, i.e.
                             the visible-hidden transformation and the
                             hidden-visible transformation use the same
                             weights.
                             
        :type corruption_level: theano.config.floatX
        :param corruption_level: Specify the level of input corruption:
                                 the probability that an input neuron's
                                 value will be fixed to 0 during computation
                                 of hidden activations.

        :type sparsity_target: float
        :param sparsity_target: The target mean for features. If set, incurs
            a sparsity penalty: the KL divergence of a unit being either off,
            or on.

        :type output_sparsity_target: float
        :param output_sparsity_target: The sparsity target for output vectors
            instead of features.

        :type L1_norm: float
        :param L1_norm: L1 regularization weight (absolute value of each
            parameter).

        :type L2_norm: float
        :param L2_norm: L2 regularization weight (quadratic value of each
            parameter).

        :type bias_decay: float
        :param bias_decay: Adds an extra L2 penalty on the bias terms.
        """
        super(DenoisingAutoencoder, self).__init__(inputs, n_in, n_out,
                                                   activation,
                                                   backward_activation,
                                                   reconstruction,
                                                   W, W_prime, b, b_prime,
                                                   tied_weights,
                                                   L1_norm, L2_norm, bias_decay,
                                                   sparsity_target,
                                                   output_sparsity_target,
                                                   rng,
                                                   theano_rng)
        self.corruption_level = corruption_level

    def mean_h_given_v(self, inputs):
        """Computes the activation of the hidden units.
        
        :type inputs: theano.tensor.TensorType
        :param inputs: Values of the visible units (i.e. rows of data).
        
        :returns: The activation on hidden units, as symbolic expression
        bound to ``inputs``.
        """
        corrupted_inputs = self.__corrupt_input(inputs)
        return self.activation(TT.dot(corrupted_inputs, self.W) + self.b)
    
    def __corrupt_input(self, inputs):
        """Randomly sets some of the inputs to zero.
        
        :type inputs: theano.tensor.TensorType
        :param inputs: Values of the visible units (i.e. rows of data).
        
        :rtype: theano.tensor.TensorType
        :returns: The inputs with some values randomly set to 0.
        """
        return self.theano_rng.binomial(size = inputs.shape, n = 1, 
                                        p = 1-self.corruption_level,
                                        dtype = theano.config.floatX) * inputs

    def _init_args_snapshot(self):
        """Saves the model in the form of an init kwarg dict, since not all
        attributes of the instance can be pickled. Upon loading, the saved
        model kwarg dict will be used as ``**kwargs`` (the ``load`` method
        is a classmethod) for an initialization of the model."""

        init_arg_dict = {
            'W' : self.W,
            'W_prime' : self.W_prime,
            'b' : self.b,
            'b_prime' : self.b_prime,
            'corruption_level' : self.corruption_level,
            'n_in' : self.n_in,
            'n_out' : self.n_out,
            'activation' : self.activation,
            'reconstruction' : self.reconstruction,
            'tied_weights' : self.tied_weights,
            'inputs' : self.inputs
            # Random number generators are ignored?
        }

        return init_arg_dict
