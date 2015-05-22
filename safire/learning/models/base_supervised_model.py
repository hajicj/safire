#!/usr/bin/env python

#
# Logistic regression using Theano
#
import pdb


import numpy
import theano
import theano.tensor as TT

import safire.utils as utils

from safire.learning.models.base_model import BaseModel
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

class BaseSupervisedModel(BaseModel):

    def __init__(self, inputs, n_in, n_out):
        """ Initialize the parameters of the logistic regression.

        A Logistic Regression layer is the end layer in classification.

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
        self.n_out = n_out
        
        self.n_layers = 1
        self.layer_shapes = [(self.n_in, self.n_out)]
        self.layers = [self] # "Simple" (one-layer) models are themselves
                             # the one and only layer the model consists of
        self.outputs = None

        self.params = []

    def error(self, y):
        """Returns the proportion of incorrectly classified instances.
        
        :type y: theano.tensor.TensorType
        :param y: A response vector.
        
        :raises: NotImplementedError()
        
        """
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
        utils.check_kwargs(kwargs, ['learning_rate', 'cost'])

        learning_rate = kwargs['learning_rate']
        bound_cost = kwargs['cost']

        updates = []
        for param in self.params:
            gradient = theano.grad(cost = bound_cost, wrt = param)
            updates.append((param, param - learning_rate * gradient))

        return updates
    
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

        It is a CLASS METHOD, which during its run actually creates
        an instance of the model. It is called as 

            >>> model_handle = ModelClass.setup(dataset, params...)

        :type data: Dataset
        :param data: The dataset on which the model will be run.

        :type model: BaseSupervisedModel
        :param model: A model instance that the setup should use.

        :type batch_size: int
        :param batch_size: how many data items will be in one minibatch
                           (the data is split to minibatches for training,
                           validation and testing)

        :type model_init_kwargs: kwargs
        :param model_init_kwargs: Various keyword arguments that get passed
                                  to the model constructor. See constructor
                                  argument documentation.
                                     
        :returns: ``ModelHandle(model, train_f, validate_f, test_f)``
                  where 'model' is the Model instance initialized during
                  :func:`setup` and the ``_func`` variables are compiled
                  theano.functions to use in a learner.                            


        """        
        index = TT.lscalar() # index of minibatch
        X = TT.matrix('X', dtype=theano.config.floatX)   # data as a matrix

        # Integer labels for classification
        #y = TT.ivector('y')  # labels as int (see data.loader.as_shared)

        # Float labels for regression
        y = TT.matrix('Y', dtype=theano.config.floatX)

        # Check for kwargs ... obsolete?
#        cls._check_init_args(model_init_kwargs)

        # Construct the model instance, or use supplied and do sanity checks.
        if model is None:
            model = cls(inputs=X, n_in=data.n_in, n_out=data.n_out,
                        **model_init_kwargs)
        else:
            # Sanity (dimensionality...) checks: 
            # - Are we passing a model of the same type as we're trying
            #   to set up?
            # - Are we passing a dataset that the model can work on?
            assert cls == type(model)
            assert model.n_in == data.n_in
            assert model.n_out == data.n_out

        # The cost to minimize during training: negative log-likelihood
        # of the training data (symbolic)
        bound_cost = model._cost(y)

        # Specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = model._training_updates(cost=bound_cost, 
                                          learning_rate=learning_rate)

        # Compile a Theano function that trains the model: returns the cost
        # (negative log-likelihood) and updates the model parameters based
        # on the rules defined in updates.
        train_model = theano.function(inputs = [X, y],
                outputs = bound_cost,
                updates = updates,
                allow_input_downcast=True)

        validate_model = theano.function(inputs = [X, y],
            outputs = model.error(y),
            allow_input_downcast=True)

        test_model = theano.function(inputs = [X, y],
            outputs = model.error(y),
            allow_input_downcast=True)

        run_model = theano.function(inputs = [X],
                            outputs = model.outputs,
                            allow_input_downcast=True)

        train_handle = ModelHandle(model, train_model)
        validate_handle = ModelHandle(model, validate_model)
        test_handle = ModelHandle(model, test_model)
        run_handle = ModelHandle(model, run_model)

        handle_dict = {'train': train_handle,
                       'validate': validate_handle,
                       'test': test_handle,
                       'run': run_handle}

        return handle_dict