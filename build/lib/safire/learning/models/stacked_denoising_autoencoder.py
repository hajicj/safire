"""
.. module:: 
    :platform: Unix
    :synopsis: ???

.. moduleauthor: Jan Hajic <hajicj@gmail.com>
"""

import theano
import theano.tensor as TT

from safire.learning.models.base_supervised_model import BaseSupervisedModel
from safire.learning.models.logistic_regression import LogisticRegression
from safire.learning.models.hidden_layer import HiddenLayer
from safire.learning.models.denoising_autoencoder import DenoisingAutoencoder
from safire.learning.interfaces.pretrained_supervised_model_handle import PretrainedSupervisedModelHandle

class StackedDenoisingAutoencoder(BaseSupervisedModel):
    """This is the base class for supervised models that
    can be pre-trained in an unsupervised fashion.
    
    The model has two aspects: one is a supervised aspect for
    classification, the other is an unsupervised aspect for pre-training
    the weights on unsupervised data so that the model better captures
    the structure of the data - it should focus on features that actually
    also describe the data well, not just that they are somehow determined
    to be useful for classification.
    
    This duality is achieved by having two stacks of layers that share
    weights. The first stack is for pre-training, the second stack is for
    the supervised (classification) task. The unsupervised layers have inputs
    from the previous feedforward layer.
    
    In the base version, the supervised portion is a classical multilayer
    perceptron with a logistic regression layer at the end. The unsupervised
    portion is composed of ``DenoisingAutoencoder``s.
    
    On :func:`setup`, the returned handle is a
    :class:`PretrainedSupervisedModelHandle` instance that as an extra field
    contains the functions for layer-wise pre-training.
    """


    def __init__(self, inputs, n_in, n_out, n_layers, n_hidden_list,
                 supervised_stack_params=None, unsupervised_stack_params=None,
                 logistic_regression_params=None):
        """Initialize a Stacked Denoising Autoencoder (SDAE).
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: thenao.tensor.TensorType
        :param inputs: symbol variable that describes the input of
        the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space
        in which the datapoints lie

        :type n_out: int
        :param n_out: number of output units - think number of classes,
        because the output units are argmaxed over to produce
        a prediction for a data point.
        
        :type n_layers: int
        :param n_layers: The number of hidden layers of the model. The
        logistic regression layer for the supervised stack is added on
        top of this, so there is ``n_layers + 1`` supervised layers in
        the end.
        
        :type n_hidden_list: list(int)
        :param n_hidden_list: A list of the hidden layer sizes. The
        logistic regression layer has then input size ``n_hidden_list[-1]``
        and output size ``n_out``.
        
        :type supervised_stack_params: list(dict)
        :param supervised_stack_params: A list of dictionaries that will
        be passed to the corresponding layer in the supervised stack as
        keyword arguments. 
        
        :type unsupervised_stack_params: list(dict)
        :param unsupervised_stack_params: A list of dictionaries that will
        be passed to the corresponding layer in the unsupervised stack as
        keyword arguments. 

        """
        # Sanity checks
        assert(n_layers == len(n_hidden_list))
        # Generate empty parameter dictionaries
        if supervised_stack_params is None:
            supervised_stack_params = [ {} for _ in xrange(n_layers)]
        if unsupervised_stack_params is None:
            unsupervised_stack_params = [ {} for _ in xrange(n_layers)]
        if logistic_regression_params is None:
            logistic_regression_params = {}
            
        assert(n_layers == len(supervised_stack_params))
        assert(n_layers == len(unsupervised_stack_params))
        
        super(StackedDenoisingAutoencoder, self).__init__(inputs, 
                                                          n_in, n_out)
    
        # Initialize intermediate variables for stack constructions
        input_sizes = [n_in] + n_hidden_list[:-1]
        output_sizes = n_hidden_list
        layer_shapes = zip(input_sizes, output_sizes)      
        
        # Initialize supervised stack
        prev_output = inputs
        self.supervised_stack = []
        for l_index, l_shape in enumerate(layer_shapes):
            # Initialize hidden layer
            current_layer = HiddenLayer(prev_output, l_shape[0], l_shape[1],
                                        **supervised_stack_params[l_index])
            self.supervised_stack.append(current_layer)
            self.params.extend(current_layer.params)
            prev_output = current_layer.outputs
    
        # Initialize logistic regression on top of supervised stack
        self.logistic_layer = LogisticRegression(
                                inputs = self.supervised_stack[-1].output,
                                n_in = self.supervised_stack[-1].n_out,
                                n_out = n_out)
        self.params.extend(self.logistic_layer.params)
        
        # Initialize unsupervised stack with shared weights. Default
        # models are autoencoders, but that's just because this class
        # needs to have something inside.
        prev_output = inputs
        self.unsupervised_stack = []
        for l_index, l_shape in enumerate(layer_shapes):
            current_layer = DenoisingAutoencoder(prev_output, 
                                        l_shape[0], l_shape[1],
                                        W = self.supervised_stack[l_index].W,
                                        b = self.supervised_stack[l_index].b,
                                        **unsupervised_stack_params[l_index])
            self.unsupervised_stack.append(current_layer)
            # Weights do not get added to layer params. (Alternatively:
            # rewrite layer params as a dictionary?)
        ##### self.params.extend([current_layer.b, current_layer.b_prime])
            # ...the previous line caused an error because the fine-tuning
            # cost doesn't include the unsupervised layer params
            prev_output = self.supervised_stack[l_index].output
            
        self.outputs = self.logistic_layer.outputs
        
    def error(self, y):
        """Computes the proportion of incorrectly classified instances.
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
        
        :returns: The proportion of incorrectly classified instances.
        
        :raises: TypeError
        """
        return self.logistic_layer.error(y)
       
    def _cost(self, y):
        """Returns the cost expression, binding the response variable for y.
        Used during setup. The cost used in the combined model is the cost
        of the supervised phase (fine-tuning the model).
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
        
        :returns: The logistic layer ``_cost`` symbolic expression bound to y.
           
        """
        return self._supervised_cost(y)
    
    def _supervised_cost(self, y):
        """Returns the cost expression, binding the response variable for y.
        Used during setup. The cost used in the combined model is the cost
        of the supervised phase (fine-tuning the model).
        
        :type y: theano.tensor.TensorType
        :param y: Corresponds to a vector that gives for each example the
                  correct label.
        
        :returns: The logistic layer ``_cost`` symbolic expression bound to y.
           
        """    
        return self.logistic_layer.negative_log_likelihood(y)
    
    def _generate_pretraining_functions(self, data, X=None, learning_rate=0.13,
                                        batch_size=500):
        """Prepares for each unsupervised layer the corresponding pre-training
        Theano function on the given data (X). This function is called during
        :func:`setup` and its outputs are appended to the resulting 
        :class:`PretrainingModelHandle`.
        
        :type X: theano.tensor.TensorType
        :param X: A theano symbolic variable that represents the input matrix.
        ``None`` by default (which creates a new variable).
        
        :type dataset: Dataset
        :param dataset: The dataset on which the pre-training should take place.
        
        :rtype: list(theano.function)
        :returns: A list of the compiled pretraining functions, one for each
        unsupervised layer (index 0 for unsupervised layer with index 0, etc.)
        """
        index = TT.lscalar()
        if X is None:
            X = TT.matrix('X')
        
        pretraining_functions = []
        for layer in self.unsupervised_stack:
            bound_cost = layer._cost(layer.inputs) 
                                  # We need this X to be the input 'processed'
                                  # by the network up to this layer.
                    # FIXME: this solution relies on the fact that the X
                    # that gets passed to this function is at the same time
                    # the symbolic variable to which the model's input is
                    # bound.
            updates = layer._training_updates(cost=bound_cost,
                                              learning_rate=learning_rate)
            pretraining_fn = theano.function(inputs = [index],
                                outputs = bound_cost,
                                updates = updates,
                                givens = {
                                    X: data.train_X_batch(index, 
                                                          batch_size)
                                })
            pretraining_functions.append(pretraining_fn)
            
        return pretraining_functions
    
    # TODO: 
    # Enable setup() to get a model instance; if an instance is given, do not
    # initialize it?
    #  - This would be useful to enable split of setup()
    #    and pretraining_setup()
    #  - This would make it possible to use the same model with already
    #    learned weights for new tasks!!! (on new datasets, etc.)
    #  - This would also necessitate that model handles share model instances
    #     - Good thing? Bad thing?
    #     - ...a step towards multi-threading? Multiple SGDs running
    #       on different parts of a dataset, each from a different
    #       model handle?
    #     - Python should take care of not deleting the instances on handle
    #       destruction...
    # Shouldn't the setup get a handle instead of a model instance and extract
    # the model instance from the handle? This would protect us from passing 
    # bare models around...
    #  - would such bare models be a bad thing? Handles should serve as access
    #    points to model training/validation/testing, but we may want to
    #    use models separately: for printing, loading & saving, etc.; a model
    #    is a fully functioning entity.
    # Separate setup() and pretraining_setup()
    @classmethod
    def setup(cls, data, model=None, batch_size=500, learning_rate=0.13, 
              data_pretrain=None, batch_size_pretrain=None, 
              learning_rate_pretrain=None, **model_init_kwargs):
        """Prepares the train_model, validate_model, test_model and
        pretrain_model functions on the given dataset and with the given 
        parameters.

        It is a CLASS METHOD, which during its run actually creates
        an instance of the model. It is called as 

            >>> model_handle = ModelClass.setup(dataset, params...)
            
        Note that the pretraining functions do not necessarily have to be
        obtained from this class method: the ``model_handle.model_instance``
        attribute of the returned handle supplies its own method to build
        the pretraining functions (and this ``setup`` uses exactly this
        method to build them).

        :type data: SupervisedDataset
        :param data: The dataset on which the model will be run.
        
        :type model: StackedDenoisingAutoencoder
        :param model: A model instance that the setup should use.

        :type batch_size: int
        :param batch_size: How many data items will be in one minibatch
                           (the data is split to minibatches for training,
                           validation and testing)
                           
        :type learning_rate: float
        :param learning_rate: A coefficient that says how much we should move
                              in the direction of the gradient during SGD
        
        :type data_pretrain: UnsupervisedDataset
        :param data_pretrain: The dataset on which you want to initialize
                              pre-training (often there is a good reason
                              to do this: there is a bigger unsupervised
                              dataset available for pre-training than
                              a supervised one for fine-tuning). If the
                              parameter is set to ``None`` (default behavior),
                              it is linked to the parameter ``data``.

        :type batch_size_pretrain: int
        :param batch_size: How many data items will be in one minibatch
                           when pre-training.
                           
        :type learning_rate_pretrain: float
        :param learning_rate: A coefficient that says how much we should move
                              in the direction of the gradient during SGD
                              in pre-training.

                              
        :type model_init_kwargs: kwargs
        :param model_init_kwargs: Various keyword arguments that get passed
                                  to the model constructor. See constructor
                                  argument documentation.
                                     
        :rtype: PretrainingModelHandle
        :returns: ``PretrainingModelHandle(model, train_f, validate_f, test_f,
                  pretraining_fs)``
                  where 'model' is the Model instance initialized during
                  :func:`setup` and the ``_f`` variables are compiled
                  theano.functions to use in a learner.                            
        """
        index = TT.lscalar() # index of minibatch
        X = TT.matrix('X')   # data as a matrix
        y = TT.ivector('y')  # labels as ints (see data.loader.as_shared)

        # Check for kwargs ... obsolete?
#        cls._check_init_args(model_init_kwargs)

        # Construct the model instance, or use supplied and do sanity checks.
        if model is None:
            model = cls(inputs=X, n_in = data.n_in, n_out = data.n_out, 
                        **model_init_kwargs)
        else:
            # Sanity (dimensionality...) checks: 
            # - Are we passing a model of the same type as we're trying
            #   to set up?
            # - Are we passing a dataset that the model can work on?
            print cls, model
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
        train_model = theano.function(inputs = [index],
                outputs = bound_cost,
                updates = updates,
                givens = {
                    X: data.train_X_batch(index, batch_size),
                    y: data.train_y_batch(index, batch_size)
                })

        # Compile a Theano function that computes the mistakes that are made
        # by the model on a minibatch of devel/test data
        validate_model = theano.function(inputs = [index],
            outputs = model.error(y),
            givens = {
                X: data.devel_X_batch(index, batch_size),
                y: data.devel_y_batch(index, batch_size)
            })

        test_model = theano.function(inputs = [index],
            outputs = model.error(y),
            givens = {
                X: data.test_X_batch(index, batch_size),
                y: data.test_y_batch(index, batch_size)
            })
        
        
        if data_pretrain is None:
            data_pretrain = data
            
        pretraining_functions = model._generate_pretraining_functions(
                                                                data_pretrain,
                                                                X,
                                                  learning_rate=learning_rate,
                                                     batch_size=batch_size)

        return PretrainedSupervisedModelHandle(model, train_model, 
                                               validate_model, test_model, 
                                               pretraining_functions)

