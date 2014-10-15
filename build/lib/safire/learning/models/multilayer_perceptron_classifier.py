import cPickle
import copy
import logging
import numpy
import theano
import theano.tensor as TT

from safire.utils import check_kwargs

from base_supervised_model import BaseSupervisedModel
from hidden_layer import HiddenLayer
from logistic_regression_classifier import LogisticRegressionClassifier


class MultilayerPerceptron(BaseSupervisedModel):
    """Basic multi-layer perceptron class.

    A multilayer perceptron is a feedforward artificial neural network
    model that has one layer or more of hidden units and nonlinear
    activations. Intermediate layers usually have as activation function
    tanh or the sigmoid function (defined here by a ``HiddenLayer`` class)
    while the top layer is a softmax layer (defined here by the class
    ``LogisticRegression``).
    """

    def __init__(self, inputs, n_in, n_out, n_layers, n_hidden_list, 
                 hidden_activation=TT.tanh, 
                 rng = numpy.random.RandomState(), 
                 L1_w = 0.0, L2_w = 0.0,
                 hidden_layer_params=None, log_regression_params=None):
        """Initialize the parameters for the multilayer perceptron.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: thenao.tensor.TensorType
        :param inputs: symbol variable that describes the input of
        the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space
        in which the datapoints lie

        :type n_out: int
        :param n_out: number of output units - again, think number
        of classes

        :type n_layers: int
        :param n_layers: The number of hidden layers of the model. The
        logistic regression layer for the supervised stack is added on
        top of this, so there is ``n_layers + 1`` supervised layers in
        the end.
        
        :type n_hidden_list: list(int)
        :param n_hidden_list: A list of the hidden layer sizes. The first
        hidden layer has the input size ``n_in`` and output size
        ``n_hidden_list[0]``.  The logistic regression layer has then 
        input size ``n_hidden_list[-1]`` and output size ``n_out``.
        
        :type hidden_activation: theano.op.Elemwise
        :param hidden_activation: The activation function to use in each
        hidden layer. This parameter is provided as a shortcut to the common
        task of changing the activation of all hidden layers; if you want more
        detailed control over the hidden layers, see the 
        ``hidden_layer_params`` attribute below.

        :type L1_w: float
        :param L1_w: weight of L1 regularization in computing the cost.

        :type L2_w: float
        :param L2_w: weight of L2 regularization in computing the cost.
        
        :type hidden_layer_params: list(dict)
        :param hidden_layer_params: If you wish, you can provide a dictionary
        of initialization args for individual hidden layers. This parameter is
        a list of such dictionaries; the dictionaries will be passed to the
        hidden layers on construction as kwargs in the form
        ``HiddenLayer(..., **hidden_layer_params[layer_index])``. Default is
        ``None``, which doesn't pass anything (so the hidden layers get all
        initialized with default ``HiddenLayer`` arguments). An exception is
        the ``hidden_activation`` parameter: if no ``hidden_layer_params`` are
        supplied, then the hidden layers are initialized with
        ``hidden_activation`` as the activation function.
        
        :type log_regression_params: dict
        :param log_regression_params: A dictionary that will be passed as
        a kwarg to the ``LogisticRegression`` layer constructor. Should only
        include parameters not otherwise supplied by the perceptron. (The
        perceptron supplies the inputs and shape of the layer.) Default is
        ``None``, which lets the layer initialize with default parameters.

        """
        # Sanity checks
        assert(n_layers == len(n_hidden_list))

        # Generate empty parameter dictionaries, take hidden_activation 
        # into account
        if hidden_layer_params is None:
            if hidden_activation is None:
                hidden_layer_params = [ {} for _ in xrange(n_layers)]
            else:
                hidden_layer_params = [ {'activation' : hidden_activation}
                                           for _ in xrange(n_layers) ]
        if log_regression_params is None:
            log_regression_params = {}
            
        assert(n_layers == len(hidden_layer_params))

        ############################

        super(MultilayerPerceptron, self).__init__(inputs, n_in, n_out)

        self.hidden_layers = []
        self.params = []

        # Initialize intermediate variables for stack constructions
        input_sizes = [n_in] + n_hidden_list[:-1]
        output_sizes = n_hidden_list
        hidden_layer_shapes = zip(input_sizes, output_sizes)
        log_layer_shape = (output_sizes[-1], n_out) 
        
        
        prev_outputs = inputs
        for l_index, l_shape in enumerate(hidden_layer_shapes):
            hparams = copy.copy(hidden_layer_params[l_index])

            # Processing necessary input arguments...
            if 'n_in' in hparams and hparams['n_in'] != l_shape[0]:
                logging.warn('Trying to initialize multilayer perceptron with mismatched'
                             'input size from params (%i) and from hidden layer shapes'
                             '(%i)' % (hparams['n_in'], l_shape[0]))
            hparams['n_in'] = l_shape[0]

            if 'n_out' in hparams and hparams['n_out'] != l_shape[1]:
                logging.warn('Trying to initialize multilayer perceptron with mismatched'
                             'output size from params (%i) and from hidden layer shapes'
                             '(%i)' % (hparams['n_out'], l_shape[1]))
            hparams['n_out'] = l_shape[0]

            # Inputs are always linked to previous outputs, so they get deleted.
            if 'inputs' in hparams:
                logging.info('Deleting inputs from hparams for layer %i; are you'
                             'really loading the model? (Where did you get the hparams?)' % l_index)
                del hparams[inputs]


            hidden_layer = HiddenLayer(inputs = prev_outputs,
                                       **hparams)
            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
            prev_outputs = hidden_layer.outputs

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        log_layer_inputs = inputs
        if n_layers > 0:
            log_layer_inputs = self.hidden_layers[-1].outputs

        # Processing logistic regression layer parameters...
        lparams = copy.copy(log_regression_params)
        if ('n_in' in lparams
                and lparams['n_in'] != log_layer_shape[0]):
            logging.warn('Log regression layer input dimension from params (%i)'
                         'and arguments (%s) doesn\'t match.' % (
                lparams['n_in'], log_layer_shape[0]))
        lparams['n_in'] = n_in

        if ('n_out' in lparams
                and lparams['n_out'] != log_layer_shape[1]):
            logging.warn('Log regression layer output dimension from params (%i)'
                         'and arguments (%s) doesn\'t match.' % (
                lparams['n_out'], log_layer_shape[1]))
        lparams['n_out'] = n_out

        if 'inputs' in lparams:
            logging.info('Deleting inputs from lparams for layer %i; are you'
                         'really loading the model? (Where did you get the lparams?)' % l_index)
            del lparams[inputs]

        self.log_regression_layer = LogisticRegressionClassifier(
            inputs=log_layer_inputs,
            **lparams)
        self.params.extend(self.log_regression_layer.params)
        
        # General model properties that need to be set.
        self.n_layers = n_layers
        self.layers = self.hidden_layers + [self.log_regression_layer]
            # Well, why shouldn't we pre-train the logistic regression layer?
        self.layer_shapes = hidden_layer_shapes + [log_layer_shape]
        self.outputs = self.log_regression_layer.outputs

        self.L1_w = L1_w
        self.L2_w = L2_w

        # We're storing the init args for save/load:
        self.n_hidden_list = n_hidden_list
        self.hidden_activation = hidden_activation
        self.hidden_layer_params = hidden_layer_params
        self.log_regression_params = log_regression_params

    def _init_args_snapshot(self):
        """Saves the model in the form of an init kwarg dict, since not all
        attributes of the instance can be pickled. Upon loading, the saved
        model kwarg dict will be used as ``**kwargs`` (the ``load`` method
        is a classmethod) for an initialization of the model."""

        # TODO: test

        hparams = [hlayer._init_args_snapshot() for hlayer in self.hidden_layers]
        lparams = self.log_regression_layer._init_args_snapshot()

        init_arg_dict = {
            'n_in' : self.n_in,
            'n_out' : self.n_out,
            'inputs' : self.inputs,
            'n_layers' : self.n_layers,
            'n_hidden_list' : self.n_hidden_list,
            'hidden_activation' : self.hidden_activation,
            'L1_w' : self.L1_w,
            'L2_w' : self.L2_w,
            'hidden_layer_params' : hparams,
            'log_regression_params' : lparams
            # Random number generators are ignored?
        }

        return init_arg_dict

    def L1(self):
        """L1 norm ; one regularization option is to enforce L1 norm to be 
        small. 
        
        :returns: Returns the number, not the function to compute it it.
        """
        return sum([abs(param).sum() for param in self.params])
        #return abs(self.hidden_layer.W).sum() + abs(self.log_regression_layer.W).sum()

    def L2(self):
        """Square of L2 norm ; one regularization option is to enforce L2 norm
        to be small.
        
        :returns: Returns the number, not the function to compute it.
        """
        return sum([(param ** 2).sum() for param in self.params])
        #return (self.hidden_layer.W ** 2).sum() + (self.log_regression_layer.W ** 2).sum()

    def negative_log_likelihood(self, y):
        """Computes the negative log likelihood of a response vector.
        Negative log likelihood of the MLP is given by the negative
        log likelihood of the output of the model, computed in the
        logistic regression layer.
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.

        :returns: The negative log likelihood of the given response under
                  the model.
        """
        return self.log_regression_layer.negative_log_likelihood(y)

    def error(self, y):
        """Computes the proportion of misclassified examples. Again, this
        is the same as the error of the logistic regression layer.
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
                  
        :returns: The proportion of incorrectly classified instances.
        
        :raises: TypeError
"""
        return self.log_regression_layer.error(y)

    def _cost(self, y):
        """Returns the cost expression, binding the response variable for y.
        Used during setup. The cost expression for this multi-layer perceptron
        model is the negative log likelihood of the response vector plus
        a regularization term.
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
        
        :returns: The negative log likelihood symbolic expression bound to y.
           
        """

        return self.negative_log_likelihood(y) \
               + self.L1_w * self.L1() + self.L2_w * self.L2()

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
        check_kwargs(kwargs, ['learning_rate', 'cost'])

        learning_rate = kwargs['learning_rate']
        bound_cost = kwargs['cost']

        updates = []
        for param in self.params:
            gradient = theano.grad(cost = bound_cost, wrt = param)
            updates.append((param, param - learning_rate * gradient))

        return updates

    ### Deprecation pending???
    @classmethod
    def _init_args(cls):
        """Returns a list of the required kwargs the class needs to be
        successfully initialized.

        Only returns args that are OVER the minimum defined in the
        BaseModel.__init__() function definition.
        
        .. warn::
        
          This method and its role is subject to change; it may also
          be removed entirely.
            
        :returns: A list of kwarg name strings.
        """
        return ['n_hidden', 'rng', 'L1_w', 'L2_w']
