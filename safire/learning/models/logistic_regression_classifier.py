#!/usr/bin/env python

#
# Logistic regression using Theano
#
import cPickle

import numpy
import theano
import theano.tensor as TT

import safire.utils as utils
from base_supervised_model import BaseSupervisedModel

class LogisticRegressionClassifier(BaseSupervisedModel):

    def __init__(self, inputs, n_in, n_out, W=None, b=None,
                 rng = numpy.random.RandomState()):
        """ Initialize the parameters of the logistic regression

        A Logistic Regression layer is the end layer in classificatio

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that descripbes the input
                       of the architecture (e.g., one minibatch of
                       input images, or output of a previous layer)


        :type n_in: int
        :param n_in: number of input units, the dimension of the space
                     in which the data points live

        :type n_out: int
        :param n_out: number of output units, the dimension of the space
                      in which the target lies

        :type W: theano.tensor.sharedvar.TensorSharedVariable
        :param W: Optionally, a shared weight matrix can be supplied.

        :type b: theano.tensor.sharedvar.TensorSharedVariable
        :param b: Optionally, a shared bias vector can be supplied.
        """
        super(LogisticRegressionClassifier, self).__init__(inputs, n_in, n_out)
        #self.n_in = n_in
        #self.n_out = n_out
        self.activation = TT.nnet.softmax
        
        if not W:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            W = self._init_weights('W', (n_in, n_out), rng)
        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(W, (n_in, n_out))

        self.W = W

        if not b:
            # initialize the biases b as a vector of n_out 0s
            b = theano.shared(value = numpy.zeros((n_out,),
                              dtype = theano.config.floatX),
                              name = 'b')

        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(b, (n_out,))

        self.b = b

        self.params = [self.W, self.b]

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = self.activation(TT.dot(inputs, self.W) + self.b)

        # Just a formality, so that the model has a common interface.
        self.outputs = self.p_y_given_x

        # compute the prediction as class whose probability is maximal
        self.y_pred = TT.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """Compute the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

          \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
          \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
             \ell (\theta=\{W,b\}, \mathcal{D})

        .. note:: 
        
          We use the mean instead of the sum so that the learning rate
          is less dependent on the batch size.

        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
                  
        :returns: The mean of the negative log-likelihood of the prediction of
                  this model under a given target distribution (see intro).
        """

        return -TT.mean(TT.log(self.p_y_given_x)[TT.arange(y.shape[0]),y])

    def error(self, y):
        """Computes the proportion of incorrectly classified instances.
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
        
        :returns: The proportion of incorrectly classified instances.
        
        :raises: TypeError
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):
            return TT.mean(TT.neq(y, self.y_pred))

        else:
            raise NotImplementedError()

    def _init_args_snapshot(self):
        """Saves the model in the form of an init kwarg dict, since not all
        attributes of the instance can be pickled. Upon loading, the saved
        model kwarg dict will be used as ``**kwargs`` (the ``load`` method
        is a classmethod) for an initialization of the model."""

        init_arg_dict = {
            'W' : self.W,
            'b' : self.b,
            'n_in' : self.n_in,
            'n_out' : self.n_out,
            'activation' : self.activation,
            'inputs' : self.inputs
            # Random number generators are ignored?
        }

        return init_arg_dict

    def _cost(self, y):
        """Returns the cost expression, binding the response variable for y.
        Used during setup. The cost used in logistic regression is the
        negative log likelihood of the response.
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
        
        :returns: The negative log likelihood symbolic expression bound to y.
           
        """
        return self.negative_log_likelihood(y)

    def _training_updates(self, **kwargs):
        """Computes the update expression for updating the model parameters
        during training.
        
        .. note::
        
          This method should only be called from the ``setup()`` class method.

        :type learning_rate: theano.config.floatX
        :param learning_rate: A coefficient by which the gradient is
                              scaled on one update step.
        
        :type cost: theano.tensor.TensorType
        :param cost: The cost expression.

        :returns: A list of ``(param, update_expr)`` tuplets that can be
                  passed directly to ``theano.function`` as the ``updates``
                  field.
        """
        utils.check_kwargs(kwargs, ['learning_rate', 'cost'])

        learning_rate = kwargs['learning_rate']
        bound_cost = kwargs['cost']

        # Problem: need symbolic 'y' for self.negative_log_likelihood(y)
        # TODO: test behavior with dummy TT.ivector symbolic variable
        g_W = TT.grad(cost = bound_cost, wrt = self.W)
        g_b = TT.grad(cost = bound_cost, wrt = self.b)
        return [(self.W, self.W - learning_rate * g_W),
                (self.b, self.b - learning_rate * g_b)]

