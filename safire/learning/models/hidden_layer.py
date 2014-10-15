import cPickle
import numpy
import theano
import theano.tensor as TT

from base_supervised_model import BaseSupervisedModel

class HiddenLayer(BaseSupervisedModel):
    def __init__(self, inputs, n_in, n_out, W = None, b = None, 
                 activation = TT.tanh, rng = numpy.random.RandomState()):
        """Typical hidden layer of a MLP: units are fully-connected
        and have sigmoidal activation function. Weight matrix W is
        of shape (n_in, n_out) and the bias vector b is of shape (n_out,).

        .. note::
            
          The nonlinearity used here by default is tanh (because 
          it is supposed to converge faster)

        Hidden unit activation is given by :math:`tanh(dot(input,W) + b)`.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.dmatrix
        :param inputs: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non-linearity to be applied in the hidden layer
        """
        super(HiddenLayer, self).__init__(inputs, n_in, n_out)
        
        self.activation = activation

        if not W:
            W = self._init_weights('W', (n_in, n_out), rng)
        else:
            self._init_param_consistency_check(W, (n_in, n_out))

        self.W = W

        # Initialize bias
        if not b:
            b = self._init_bias('b', n_out, rng)
        else:
            self._init_param_consistency_check(b, (n_out, ))

        self.b = b

        # The layer_shapes member has already been initialized in super.init,
        # but we're doing it here explicitly anyway
        self.n_layers = 1
        self.layer_shapes = [(n_in, n_out)]
        self.layers = [self]
        self.outputs = self.activation(TT.dot(self.inputs, self.W) + self.b)

        # Parameters of the model
        self.params = [self.W, self.b]

    def _get_hidden_values(self, inputs=None):
        """Computes the activation of the hidden units.

        :type inputs: theano.tensor.TensorType
        :param inputs: Values of the visible units (i.e. rows of data).

        :returns: The activation on hidden units, as symbolic expression
        bound to ``inputs``.
        """
        if inputs:
            return self.activation(TT.dot(inputs, self.W) + self.b)
        else:
            return self.activation(TT.dot(self.inputs, self.W) + self.b)

    def mean_squared_error(self, y):
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
        :param y: Corresponds to a matrix of the desired output vectors.

        :returns: The mean of the squared error of the prediction against the
            true values.
        """
        return TT.mean((self._get_hidden_values(self.inputs) - y) ** 2)

    def _cost(self, y):
        """Returns the proportion of incorrectly classified instances.
        
        .. warn::
        
          The hidden layer is not intended to supply error to a model. It is
          an internal layer; error is computed at the output layer.
        
        :type y: theano.tensor.TensorType
        :param y: A response vector.
        
        :raises: NotImplementedError
        """
        return self.mean_squared_error(y)

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

    def error(self, y):
        """Returns the proportion of incorrectly classified instances.
        
        .. warn::
        
          The hidden layer is not intended to supply error to a model. It is
          an internal layer; error is computed at the output layer.
        
        :type y: theano.tensor.TensorType
        :param y: A response vector.
        
        :raises: NotImplementedError
        """
        if y.ndim != self.outputs.ndim:
            raise TypeError('y should have the same shape as self.outputs', ('y', y.type, '/ outputs', self.outputs.type))

        return self.mean_squared_error(y)

    def _training_updates(self, **kwargs):
        """Returns the update expression for updating the model parameters
        during training. The formula for updating an argument is
            
        .. math:
            
          \theta^{(k+1)} = \theta^{(k)} - learning\_rate * \frac{\partial cost}{\partial \theta} 

        Expects a 'learning_rate' and 'cost' kwarg.
            
        :type learning_rate: theano.config.floatX
        :param learning_rate: The learning rate for parameter updates.
                              Scaling of the step in the direction of the
                              gradient.
                                  
        :type cost: theano.tensor.TensorType
        :param cost: The cost function of which we are computing
                         the gradient.
                         
        :raises: NotImplementedError
        """

        raise NotImplementedError('Hidden layer of a perceptron by itself cannot train (only output layers can train).')


    @classmethod
    def _init_args(cls):
        """Returns a list of the required kwargs the class needs to be
        successfully initialized.

        Only returns args that are OVER the minimum defined in the
        BaseModel.__init__() function definition.
 
        .. warn::
        
          This method and its role is subject to change; it may also
          be removed entirely.
                        
        :returns: A list of strings.
        """
        return ['rng', 'activation']
