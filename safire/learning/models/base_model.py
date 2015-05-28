#!/usr/bin/env python

#
# Logistic regression using Theano
#
import codecs
import cPickle
import logging

import numpy
import theano
import theano.tensor as TT

import safire.utils
from safire.learning.interfaces.model_handle import ModelHandle

# TODO: Rewrite setup() as instance method??? Is it even possible?
#       Why: so that a learner will only get a model instance, not
#       a model class. (A model class can be passed, of course. However,
#       multiple learners might want to update some parameters in parallel.
#
#       !!!! BUT: we can run Model.setup() outside the learner, so the
#                 train_, devel_ and test_ functions refer to the same
#                 shared object, and pass these functions as parameters
#                 to learners, not the Model classes.
#
#       - this actually reduces coupling between Learner and Model classes,
#         since a learner may use validation completely unrelated to training
#         or the test data can be perfectly hidden from the model on setup().
#         ...which presupposes that there is Model.setup_train(), setup_test(),
#            setup_devel() which can be called separately with only the given
#            data.

class BaseModel(object):

    def __init__(self, inputs, n_in, **model_init_kwargs):
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
        """

        self.inputs = inputs
        self.n_in = n_in
        self.n_out = 0
        
        # Deprecate? ... no no no, will use this in linking unsupervised
        # models.
        self.layer_shapes = [(self.n_in, self.n_out)]
        self.n_layers = 0
        self.layers = []
        self.outputs = None

        self.params = []


    def error(self, y):
        """Returns some measure of model error, to be used in validation
        and testing to check on training progress.
        
        :type y: theano.tensor.TensorType
        :param y: A response vector in supervised models -- and something
                  else in unsupervised ones (most often the data).
        
        :raises: NotImplementedError()
        
        """
        raise NotImplementedError()

    def save(self, filename, protocol=-1):
        """Saves the model in the form of an init kwarg dict, since not all
        attributes of the instance can be pickled. Upon loading, the saved
        model kwarg dict will be used as ``**kwargs`` (the ``load`` method
        is a classmethod) for an initialization of the model.

        **The _init_args_snapshot() method needs to be overriden by each model
        to provide a dict of init args.**

        :type filename: str
        :param filename: The name of the file to which the model parameters
                         should be stored.

        :type protocol: int
        :param protocol: The cPickle protocol to use (default: -1)
        """

        pickling_object = self._export_pickleable_object()

        with open(filename, 'wb') as pickle_handle:
            cPickle.dump(pickling_object, pickle_handle, protocol)

    def save_layer(self, layer, filename, protocol=-1):
        """Saves one layer from the current model to the given file.

        :type layer: int
        :param layer: The index of the layer to save (starting from 0).

        :type filename: str
        :param filename: The file to which to save the layer model.

        :type protocol: int
        :param protocol: The pickling protocol to use (default: highest).
        """
        assert layer <= len(self.layers), "Only %i layers available (requested: %i)" % (layer, len(self.layers))

        self.layers[layer].save(filename, protocol)

    def _export_pickleable_object(self):
        """Builds the object that will actually be pickled. Relies on the
        ``_init_args_snapshot()`` method being implemented and adds the class
        object, so that the model can be reconstructed without knowing its
        type in advance."""
        init_arg_dict = self._init_args_snapshot()
        model_class = self.__class__

        pickling_object = { 'init_args': init_arg_dict,
                            'class': model_class }

        return pickling_object

    def _init_args_snapshot(self):
        """Returns the list of init args so that the current model can
        be exactly reconstructed using these arguments."""
        raise NotImplementedError()


    def _cost(self, y):
        """Returns the cost expression, binding the response variable for y.
        Used during setup.

        :type y: theano.tensor.vector
        :param y: The response variable against which the cost is computed
           
        :raises: NotImplementedError()
        """
        raise NotImplementedError()

    def _training_updates(self, **kwargs):
        """Returns the update expression for updating the model parameters
        during training. The formula for updating an argument is
            
        .. math:
            
           \theta^{(k+1)} = \theta^{(k)} - learning\_rate * \frac{\partial cost}{\partial \theta} 

        Expects a 'learning_rate' and 'cost' kwarg.
            
        :type learning_rate: theano.config.floatX
        :param learning_rate: The learning rate for parameter updates.
                                  
        :type cost: theano.tensor.TensorType
        :param cost: The cost function of which we are computing
                     the gradient.
                         
        :returns: A list of pairs (parameter, update_expression), to
                  be passed directly to ``theano.function`` as the
                  ``updates`` parameter.
        """
        raise NotImplementedError()
    
    def _init_weights(self, name, shape, rng = numpy.random.RandomState()):
        """Initialize a weights parameter. The weights are randomly
        drawn from a uniform distribution over an interval specified
        in the :func:`__init_weights_bounds` method, which
        takes care of adjusting this interval for various activation
        functions.
        
        :type name: str
        :param name: The name of the weights parameter. Used in naming
                     the created shared variable.
        
        :type shape: tuple(int,int)
        :param shape: The shape of the weights matrix. The first number
                      is typically the number of input neurons, the second
                      number is the number of output neurons of the layer.
        
        :type rng: numpy.random.RandomState
        :param rng: A random number generator from which the weights will
                    be drawn.
                    
        :rtype: theano.tensor.sharedvar.TheanoSharedVariable
        :returns: The initialized weights matrix.
        """
        lbound, rbound = self.__init_weights_bounds(self.activation)
        W = safire.utils.random_shared_var(name, shape, lbound, rbound, rng)
        return W
    
    def _init_bias(self, name, length, rng = numpy.random.RandomState()):
        """Initialize a bias parameter. The bias values are set to
        0.001 (default behavior to avoid some NaN's in first iteration of
        training).
        
        :type name: str
        :param name: The name of the bias parameter. Used in naming
                     the created shared variable.
        
        :type length: int
        :param length: The size of the bias vector (number of output neurons
                       of the model).
        
        :type rng: numpy.random.RandomState
        :param rng: A random number generator from which the weights will
                    be drawn. Not currently used in the base model.
                    
        :rtype: theano.tensor.sharedvar.TheanoSharedVariable
        :returns: The initialized bias vector.
        """
        b = theano.shared(value=numpy.zeros((length,),
                                            dtype=theano.config.floatX) + 0.001,
                          name=name)
        return b
        

    def _init_param_consistency_check(self, param, shape):
        """Checks the given parameter for type 
        (theano.tensor.sharedvar.TensorSharedVariable) and given shape.
        If check fails, raises a ``TypeError``.

        :type param: theano.tensor.sharedvar.TensorSharedVariable
        :param param: A parameter candidate. The type is the intended
                      type; this function checks that it is so.

        :type shape: tuple
        :param shape: A tuple of integers that specifies the expected
                      shape of param.

        :raises: TypeError
        """
        # This method is name-mangled - no one should *ever* want to access it
        # anywhere but the model __init__ for checking supplied parameter
        # consistency.
        if not (isinstance(param, theano.tensor.sharedvar.TensorSharedVariable)
            or isinstance(param, theano.sandbox.cuda.var.CudaNdarraySharedVariable)):
            raise TypeError('Init param check: expects theano shared variable as constructor parameter argument, not %s' % str(type(param)))

        if not param.get_value(borrow=True).shape == shape:
            raise TypeError('Constructor parameter argument has incorrect shape %s, expects %s.' % (str(param.get_value(borrow=True).shape), str(shape)))
        
    def __init_weights_bounds(self, activation):
        """Returns parameter initialization bounds for the given activation
        function. Various models may want to override this, if there are
        good mathematical reasons to do it.
        
        Currently defaults to ``TT.nnet.tanh`` values and can deal with
        ``TT.nnet.sigmoid``.
        
        .. warn:
        
          This method is implemented in the
        
        :type activation: any
        :param activation: An activation function.
        
        :rtype: tuple(theano.config.floatX, theano.config.floatX)
        :returns: A tuple ``(lbound, rbound)`` that specifies the bounds
                  of the interval from which parameter weights should be
                  initialized. 
        """
#        raise NotImplementedError('BaseModel does not have sufficient ' 
#            + 'information about the properties of the model to initialize '
#            + 'parameter bounds.')
# NOTE: with fixing unsupervised models to contain both n_hidden and n_in,
#       we have a base model that does contain all the necessary information.
        if activation == safire.utils.ReLU:
            lbound = 0.00001
            rbound = 0.5
            return (lbound, rbound)

        n_neurons = self.n_in * self.n_out
        lbound = -numpy.sqrt(6. / n_neurons)
        rbound = numpy.sqrt(6. / n_neurons)
        
        if activation == TT.nnet.sigmoid:
            lbound *= 4
            rbound *= 4
            
        return (lbound, rbound)
    
    def __initialize_parameter(self, name, shape, lbound, rbound, rng=numpy.random.RandomState):
        """Initializes a parameter according to the specification and
        appends it to the ``params`` attribute of the model. Low-level
        method, just a wrapper for ``utils.random_shared_var`` that appends
        the result to the correct place.
        
        :type name: str
        :param name: The name of the parameter. Used in naming
                     the created shared variable.

        :type shape: 
        :param shape: The desired shape of the parameter.

        :type lbound: theano.config.floatX
        :param lbound: The lower bound of the values with which
                       to initialize the parameter.

        :type rbound: theano.config.floatX
        :param rbound: The upper bound of the values with which
                       to initialize the shared variable.

        :type rng: numpy.random.RandomState
        :param rng: Optionaly supply a random number generator.
                    If ``None`` (default), creates a new one with
                    a random seed.             

        """
        self.params.append(safire.utils.random_shared_var(name, shape, lbound,
                                                   rbound, rng))
        

    def __get_param_serialization_map(self):
        """Provides a list of references to theano shared variable params
        that should be saved when saving the model. The important thing is
        the order in which the parameters are returned: this function ensures
        that on loading, the shared variable values are assigned to the
        appropriate shared variables.

        :rtype: list
        :returns: A list of references to the model's parameters
                  that are loaded from a saved model parameter file.
                  This does not necessarily mean simply a (possibly
                  re-ordered) copy of the models's ``params`` attribute;
                  for instance models with pre-training will pickle
                  also unsupervised layer biases, which are not linked
                  to the supervised layers' biases that are in the
                  pretrained model's ``params`` list. 
        """
        raise NotImplementedError()

    @classmethod
    def _init_args(cls):
        """Returns a list of the required kwargs the class needs to be
        successfully initialized.

        Only returns args that are OVER the minimum defined in the
        BaseModel.__init__() function definition.
        
        :returns: Empty list.
        """
        return []

    @classmethod
    def _check_init_args(cls, args):
        """Raises a TypeError if all _init_args() are not present in the given
        args dictionary (will actually take any iterable, but there's no
        point in doing this with anything else but **kwargs passed to 
        _setup()...)
        
        :type args: dict
        :param args: The kwarg dictionary in which to look for required args.
            
        :raises: TypeError
        """
        required_kwargs = cls._init_args()
        for arg in required_kwargs:
            if arg not in args:
                raise TypeError("Arg \'%s\' required by model class %s not available" % (arg, str(cls)))

    @classmethod
    def setup(cls, data, model=None, batch_size=500, learning_rate=0.13, 
              **model_init_kwargs):
        """Prepares the train_model, validate_model and test_model methods
        on the given dataset and with the given parameters.
        
        .. Note::
        
          In this base class, all this method does is raise 
          a ``NotImplementedError``. To successfully run the setup, we need
          to know whether the model is in an unsupervised or supervised (or
          some other mixed) setting.

        It is a CLASS METHOD, which during its run actually creates
        an instance of the model. It is called as 

            >>> model_handle = ModelClass.setup(dataset, params...)

        :type data: Dataset
        :param data: The dataset on which the model will be run.

        :type model: BaseModel
        :param model: A model instance that the setup should use.

        :type batch_size: int
        :param batch_size: how many data items will be in one minibatch
                           (the data is split to minibatches for training,
                           validation and testing)

        :type model_init_kwargs: kwargs
        :param model_init_kwargs: Various keyword arguments that get passed
                                  to the model constructor. See constructor
                                  argument documentation.
                                            
        :raises: NotImplementedError                            
        """        
        raise NotImplementedError('Cannot call setup() of BaseModel.')

    @classmethod
    def load(cls, filename, load_any=False):
        """Loads a saved model.

        When explicitly loading a model, we should know the class in advance,
        so it is not necessary to process the ``class`` member of the pickled
        dict. However, at least a warning is given if the pickled class and
        load()-calling class do not agree.

        :type load_any: bool
        :param load_any: If set to ``True``, will return an object of the type
            saved in the ``class`` member of the pickled dict. If False, will
            load the caller model class. Set to True if you want a factory-like
            behavior of BaseModel.load() (Useful for situations where you do
            not know the class in advance, such as resuming in learners.)
        """

        pickled_object = cls._load_pickleable_object(filename)

        if cls != pickled_object['class']:
            logging.warn('Caller class (%s) and pickled class (%s) do not match.' %
                         (cls, pickled_object['class']))
        if load_any:
            loader_class = pickled_object['class']
        else:
            loader_class = cls

        init_arg_dict = pickled_object['init_args']

        obj = loader_class(**init_arg_dict)

        return obj

    @classmethod
    def _load_pickleable_object(cls, filename):
        """Loads only the pickleable object and doesn't initialize the model.
        Useful for loading models without knowing their type first - you can
        then initialize them using the ``class`` member of the pickled
        object."""
        with open(filename, 'rb') as unpickle_handle:
            pickled_object = cPickle.load(unpickle_handle)

        return pickled_object

    def __getstate__(self):
        logging.debug('=== Pickling {0}:'.format(self.__class__.__name__))
        for k in self.__dict__:
            logging.debug('{0}: type {1}'.format(k, type(self.__dict__[k])))
        return self.__dict__