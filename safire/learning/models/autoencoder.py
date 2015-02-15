"""
.. module:: 
    :platform: Unix
    :synopsis: This module contains the Autoencoder class definition.

.. moduleauthor: Jan Hajic <hajicj@gmail.com>
"""
import cPickle

import numpy
import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

from safire.learning.models.base_unsupervised_model import BaseUnsupervisedModel


class Autoencoder(BaseUnsupervisedModel):
    """The Autoencoder model. 
    
    Optimizes reconstruction error with a cross-entropy loss function.
    
    """
    def __init__(self, inputs, n_in, n_out=100,
                 activation=TT.nnet.sigmoid,
                 backward_activation=TT.nnet.sigmoid,
                 reconstruction='cross-entropy',
                 W=None, W_prime=None, b=None, b_prime=None, 
                 tied_weights=True,
                 L1_norm=0.0, L2_norm=0.0, bias_decay=0.0,
                 sparsity_target=None, output_sparsity_target=None,
                 rng=numpy.random.RandomState(),
                 theano_rng=None):
        """Initialize the parameters of the Autoencoder.
        An Autoencoder is an unsupervised model that tries to minimize
        reconstruction error on input.

        :type inputs: theano.tensor.TensorType
        :param inputs: Symbolic variable that descripbes the input
                       of the architecture (e.g., one minibatch of
                       input images, or output of a previous layer)


        :type n_in: int
        :param n_in: Number of input units, the dimension of the space
                     in which the data points live

        :type n_out: int
        :param n_out: The number of output/hidden units.
        
        :type activation: theano.tensor.elemwise.Elemwise
        :param activation: The nonlinearity applied at neuron
                           output.

        :type reconstruction: str
        :param reconstruction: Which reconstruction cost to use. Accepts
            ``cross-entropy`` and ``mse`` (for Mean Squared Error).
                      
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
        super(Autoencoder, self).__init__(inputs, n_in, n_out)

        self.activation = activation
        self.backward_activation = backward_activation
        self.tied_weights = tied_weights # Bookkeeping, basically.
        self.reconstruction = reconstruction

        self.L1_norm = L1_norm
        self.L2_norm = L2_norm
        self.bias_decay = bias_decay
        self.sparsity_target = sparsity_target
        self.output_sparsity_target = output_sparsity_target
         
        if not W:
            W = self._init_weights('W', (n_in, n_out), rng)
        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(W, (n_in, n_out))

        self.W = W

        # W_prime needs different behavior for tied_weights=True/False
        self.W_prime = None
        
        if tied_weights is True:
            self.W_prime = self.W.T
        
        else:
            if not W_prime:
                W_prime = self._init_weights('W_prime', (n_out, n_in), rng)
            else: # Check for consistency of supplied weights 
                self._init_param_consistency_check(W_prime, (n_out, n_in))
            self.W_prime = W_prime

        if not b:
            b = self._init_bias('b', n_out, rng)

        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(b, (n_out,))

        self.b = b
        
        if not b_prime:
            # initialize the biases b as a vector of n_out 0s
            b_prime = self._init_bias('b_prime', n_in, rng)
        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(b_prime, (n_in,))

        self.b_prime = b_prime
        
        # Different params for tied weights!
        # This will be difficult to put in a general method.
        self.params = [self.W, self.b, self.b_prime]
        if not self.tied_weights:
            self.params.append(self.W_prime)

        # Compatibility of interface
        self.outputs = self.activation(TT.dot(inputs, self.W) + self.b)

        if theano_rng is None:
            theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.theano_rng = theano_rng

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
            'n_in' : self.n_in,
            'n_out' : self.n_out,
            'activation' : self.activation,
            'tied_weights' : self.tied_weights,
            'inputs' : self.inputs,
            'reconstruction' : self.reconstruction,
            'L1_norm': self.L1_norm,
            'L2_norm': self.L2_norm,
            'bias_decay': self.bias_decay,
            'sparsity_target': self.sparsity_target,
            'output_sparsity_target': self.output_sparsity_target,
            # Random number generators are ignored?
        }

        return init_arg_dict

    def mean_h_given_v(self, inputs):
        """Computes the activation of the hidden units.
        
        :type inputs: theano.tensor.TensorType
        :param inputs: Values of the visible units (i.e. rows of data).
        
        :returns: The activation on hidden units, as symbolic expression
        bound to ``inputs``.
        """
        return self.activation(TT.dot(inputs, self.W) + self.b)
    
    def mean_v_given_h(self, hidden_values):
        """Computes the activation of the visible units on reconstruction.

        :type hidden_values: theano.tensor.TensorType
        :param hidden_values: Values of the hidden units.
        
        :returns: The activation on visible units, as symbolic expression
        bound to ``hidden_values``. This is the reconstructed activation.
        
        """
        return self.backward_activation(TT.dot(hidden_values, self.W_prime) + self.b_prime)

    def sample_v_given_h(self, hidden):
        """Samples the visible layer given the hidden layer."""
        mean_v = self.activation(TT.dot(hidden, self.W_prime) + self.b_prime)
        sample_v = self.theano_rng.binomial(size=mean_v.shape,
                                            n=1, p=mean_v,
                                            dtype=theano.config.floatX)
        return sample_v

    def sample_h_given_v(self, visible):
        """Samples the hidden layer given the visible layer."""
        mean_h = self.backward_activation(TT.dot(visible, self.W) + self.b)
        sample_h = self.theano_rng.binomial(size=mean_h.shape,
                                            n=1, p=mean_h,
                                            dtype=theano.config.floatX)
        return sample_h

    def sample_vhv(self, visible):
        """Performs one Gibbs sampling step from visible to visible layer."""
        return self.sample_v_given_h(self.sample_h_given_v(visible))

    def sample_hvh(self, hidden):
        """Performs one Gibbs sampling step from hidden to hidden layer."""
        return self.sample_h_given_v(self.sample_v_given_h(hidden))

    def _reconstruction_cross_entropy(self, X):
        """Computes the reconstruction cross-entropy on X.
        
        :type X: theano.tensor.TensorType
        :param X: A training batch. In comparison to a supervised model,
                  which computes cost on some response vector, the
                  unsupervised model has to compute cost on the inputs.
        
        :returns: The reconstruction cross-entropy on X, as a number.
        """
        activation_hidden = self.mean_h_given_v(X)
        activation_visible = self.mean_v_given_h(activation_hidden)
        return -TT.sum(X * TT.log(activation_visible) + (1 - X)
                       * TT.log(1 - activation_visible), axis=1)
            # A -TT.sum(...) here; should the negative really be
            # there or not?

    def _reconstruction_squared_error(self, X):
        """Computes the reconstruction squared error on X.

        :type X: theano.tensor.TensorType
        :param X: A training batch. In comparison to a supervised model,
                  which computes cost on some response vector, the
                  unsupervised model has to compute cost on the inputs.

        :returns: The reconstruction squared error on X, as a number.
        """
        activation_hidden = self.mean_h_given_v(X)
        activation_visible = self.mean_v_given_h(activation_hidden)

        return (activation_visible - X) ** 2

    def _reconstruction_hypercubic_exploded_error(self, X):
        """Computes the reconstruction hypercubic (to the power of 4) error
        and multiplies it by a significant number."""

        activation_hidden = self.mean_h_given_v(X)
        activation_visible = self.mean_v_given_h(activation_hidden)

        return 10.0 * ((activation_visible - X) ** 10)

    def error(self, X):
        """Returns the mean reconstruction cross-entropy on X.
        This is the same number which is used for model cost to optimize
        in gradient descent, since without gold-standard data, we have
        nothing to really compute any error on. So, validation and
        testing can use this function in guarding against overfitting.

        :type X: theano.tensor.TensorType
        :param X: A training batch. In comparison to a supervised model,
                  which computes cost on some response vector, the
                  unsupervised model has to compute cost on the inputs.

        :returns: The reconstruction cross-entropy (as a number)
        """
        return self._cost(X)
        # if self.reconstruction == 'cross-entropy':
        #     return TT.mean(self._reconstruction_cross_entropy(X))
        # elif self.reconstruction == 'mse':
        #     return TT.mean(self._reconstruction_squared_error(X))
        # elif self.reconstruction == 'exaggerated-mse':
        #     return TT.mean(self._reconstruction_hypercubic_exploded_error(X))
        # else:
        #     raise ValueError('Invalid reconstruction set! %s' % self.reconstruction)

    def _cost(self, X):
        """Returns the mean reconstruction cross-entropy on X.
        This is the same number which is used for model error.
        
        :type X: theano.tensor.TensorType
        :param X: A training batch. In comparison to a supervised model,
                  which computes cost on some response vector, the
                  unsupervised model has to compute cost on the inputs.

        :returns: The reconstruction cross-entropy (as a number)
        """
        if self.reconstruction == 'cross-entropy':
            cost = TT.mean(self._reconstruction_cross_entropy(X))
        elif self.reconstruction == 'mse':
            cost = TT.mean(self._reconstruction_squared_error(X))
        elif self.reconstruction == 'exaggerated-mse':
            cost = TT.mean(self._reconstruction_hypercubic_exploded_error(X))
        else:
            raise ValueError('Invalid reconstruction set! %s' % self.reconstruction)

        if self.L1_norm != 0.0:
            cost += (TT.sqrt(TT.sum(self.W ** 2))
                     + TT.sqrt(TT.sum(self.W_prime ** 2))
                     + TT.sqrt(TT.sum(self.b ** 2))
                     + TT.sqrt(TT.sum(self.b_prime ** 2))) * self.L1_norm

        if self.L2_norm != 0.0:
            cost += (TT.sum(self.W ** 2) + TT.sum(self.W_prime ** 2)
                     + TT.sum(self.b ** 2) + TT.sum(self.b_prime ** 2)) \
                    * self.L2_norm

        if self.bias_decay != 0.0:
            cost += (TT.sum(self.b ** 2) + TT.sum(self.b_prime ** 2)) \
                    * self.bias_decay

        if self.sparsity_target is not None:
            cost += self._sparsity_cross_entropy(X)

        if self.output_sparsity_target is not None:
            print 'Setting output sparsity target: {0}'.format(self.output_sparsity_target)
            cost += self._output_sparsity_cross_entropy(X)

        return cost

    def _sparsity_cross_entropy(self, X):
        """
        Computes the KL divergence of distribution of the sparsity target
        w.r.t. mean activation of each hidden neuron.

        :param X: The input data batch.

        :return: The KL-divergence... (see desc.)
        """
        mean_act = TT.mean(self.activation(TT.dot(X, self.W) + self.b),
                           axis=0)
        rho_term = mean_act * TT.log(mean_act / self.sparsity_target)
        mean_act_compl = 1.0 - mean_act
        neg_rho_term = mean_act_compl * TT.log(mean_act_compl /
                                               (1.0 - self.sparsity_target))
        kl_divergence = TT.sum(rho_term + neg_rho_term)

        return kl_divergence

    def _output_sparsity_cross_entropy(self, X):
        """
        Computes the KL divergence of distribution of the sparsity target
        w.r.t. mean activation of each hidden neuron.

        :param X: The input data batch.

        :return: The KL-divergence... (see desc.)
        """
        mean_act = TT.mean(self.activation(TT.dot(X, self.W) + self.b),
                           axis=1)
        rho_term = mean_act * TT.log(mean_act / self.output_sparsity_target)
        mean_act_compl = 1.0 - mean_act
        neg_rho_term = mean_act_compl * TT.log(mean_act_compl /
                                               (1.0 - self.output_sparsity_target))
        kl_divergence = TT.sum(rho_term + neg_rho_term)

        return kl_divergence

    @classmethod
    def _init_args(cls): # This method will get obsolete.
        """Returns a list of the required kwargs the class needs to be
        successfully initialized.

        Only returns args that are OVER the minimum defined in the
        BaseModel.__init__() function definition.
 
        .. warn::
        
          This method and its role is subject to change; it may also
          be removed entirely.
                        
        :returns: A list of strings: ``['n_out', 'activation', 'rng']``
        """
        return ['n_out', 'activation', 'rng', 'reconstruction']
       
