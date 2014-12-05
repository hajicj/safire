#!/usr/bin/env python

#
# Logistic regression using Theano
#

import numpy
import theano
import theano.tensor as TT

import safire
from safire.utils import check_kwargs
from safire.learning.models.base_model import BaseModel
from safire.learning.interfaces.model_handle import ModelHandle
from safire.learning.interfaces.pretraining_model_handle import PretrainingModelHandle
from safire.learning.updaters import StandardSGDUpdater


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


class BaseUnsupervisedModel(BaseModel):

    def __init__(self, inputs, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        A Logistic Regression layer is the end layer in classificatio

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that descripbes the input
                       of the architecture (e.g., one minibatch of
                       input images, or output of a previous layer)


        :type n_in: int
        :param n_in: number of input units, the dimension of the space
                     in which the data points live

        :type n_hidden: int
        :param n_hidden: number of hidden units
        """

        self.inputs = inputs
        self.n_in = n_in

        self.n_out = n_out

        self.n_hidden = n_out # This is a technical bypass of the n_hidden
                              # vs. n_out dichotomy in base models. The naming
                              # scheme properly reflects the difference in
                              # meaning between supervised and unsupervised
                              # networks, but there are methods in parameter
                              # initialization that need to know the total
                              # number of neurons in the network and these
                              # methods definitely belong to the base model
                              # (they would be reimplemented over and over).

                    # TODO: this dichotomy leads to a different size tracking
                    # mechanism for multi-layer networks. n_in and n_out
                    # should describe the expected input and output layer
                    # sizes, which in the case of more complicated networks
                    # (recurrent?) may become quite complex, and a separate
                    # member (layer_sizes?) should hold the layer sizes.

        self.n_layers = 1
        self.layer_shapes = [(self.n_in, self.n_out)]
        self.layers = [self]
        self.outputs = None # Models have to override outputs!

        self.params = []


    def error(self, X):
        """Returns the proportion of incorrectly classified instances.

        :type X: theano.tensor.TensorType
        :param X: A response vector. Note that
                  in an unsupervised model, this is *not* the response like
                  in the supervised model -- more likely it's the input data
                  themselves.

        :raises: NotImplementedError()

        """
        raise NotImplementedError()

    def _cost(self, X):
        """Returns the cost expression, binding the response variable for X.
        Used during setup.

        :type X: theano.tensor.vector
        :param X: The variable against which the cost is computed. Note that
                  in an unsupervised model, this is *not* the response like
                  in the supervised model -- more likely it's the input data
                  themselves.

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
        #check_kwargs(kwargs, ['learning_rate', 'cost'])

        updater = kwargs['updater']
        #learning_rate = kwargs['learning_rate']
        bound_cost = kwargs['cost']

        gradients = []
        for param in self.params:
            gradients.append(theano.grad(cost=bound_cost, wrt=param))

        updates = updater(self.params, gradients)
        # updates = []
        # for param, gradient in zip(self.params, gradients):
        #     updates.append((param, param - learning_rate * gradient))

        return updates

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
    def link(cls, model_instance, layer_index, **model_init_args):
        """Constructs a model for pretraining the ``model_instance`` layer
        given by ``layer_index``. Does NOT provide the training function,
        only constructs the model with all links correctly initialized.

        :type model_instance: BaseSupervisedModel
        :param model: The model instance which should be pre-trained.

        :type layer_index: int
        :param layer_index: Which layer of ``model_instance`` to link to.
        Starting with 0, the output layer of the model would be index
        ``model_instance.n_layers - 1``.

        :type model_init_args: kwargs
        :param model_init_args: Various keyword arguments passed to
        pretraining model constructor.

        :rtype: BaseUnsupervisedModel
        :returns: An instance of ``cls`` linked to the given layer of
        the given supervised model.
        """

        # Sanity checks need to go here
        # Only allow layers that exist in the model, but allows negative
        # indexing (-1 as the last layer...)
        assert (isinstance(model_instance, BaseModel))
        assert ((layer_index < model_instance.n_layers)
                and (abs(layer_index) <= model_instance.n_layers))

        link_layer = model_instance.layers[layer_index]
        pretraining_layer = cls(inputs = link_layer.inputs,
                                n_in = link_layer.n_in,
                                n_out = link_layer.n_out,
                                W = link_layer.W, # EAFP here...
                                b = link_layer.b, # ...yeah. Screw-up on RBMs.
                                **model_init_args)

        return pretraining_layer

    @classmethod
    def setup_as_pretraining(cls, data, supervised_model_instance,
                             linked_layer_index,
                             batch_size=500, learning_rate=0.13,
                             **model_init_args):
        """Links a model instance to the ``link_layer_index``-th layer
        of ``supervised_model_instance`` for pre-training and generates
        the pretraining function. Returns a ``PretrainingModelHandle``
        that contains the pretraining model instance and the pretraining
        function.

        :type data: Dataset
        :param data: The dataset on which pre-training should work.

        :type model_instance: BaseSupervisedModel
        :param model: The model instance which should be pre-trained.

        :type linked_layer_index: int
        :param linked_layer_index: Which layer of ``model_instance`` to
        link to. Starting with 0, the output layer of the model would be
        index ``model_instance.n_layers - 1``.

        :type batch_size: int
        :param batch_size: how many data items will be in one minibatch
        (the data is split to minibatches for training,
        validation and testing)

        :type learning_rate: theano.config.floatX
        :param learning_rate: How fast will the model move in the direction
        of the gradient.

        :type model_init_args: kwargs
        :param model_init_args: Various keyword arguments passed to
        pretraining model constructor.

        """

        # TODO: sanity checks

        model = cls.link(supervised_model_instance, linked_layer_index,
                         **model_init_args)

        bound_cost = model._cost(model.inputs)
        updates = model._training_updates(cost = bound_cost,
                                          learning_rate = learning_rate)

        # Notice the trick in inputs = []: we link the data to the
        # supervised_model_instance's inputs, so that the data runs
        # through the previous layers first and gets correctly transformed.
        batch_index = TT.lscalar('batch_index')
        pretrain_model = theano.function(inputs = [supervised_model_instance.inputs],
                                    outputs = bound_cost,
                                    updates = updates,
                                    allow_input_downcast=True)

        return PretrainingModelHandle(model, pretrain_model)


    @classmethod
    def setup(cls, data, model=None, batch_size=500, learning_rate=0.13,
              heavy_debug=False, **model_init_kwargs):
        """Prepares the train_model, validate_model and test_model methods
        on the given dataset and with the given parameters.

        It is a CLASS METHOD, which during its run actually creates
        an instance of the model. It is called as

            >>> model_handle = ModelClass.setup(dataset, params...)

        The dataset is normally expected to provide information about the input
        and output dimension of the model. However, this is not true in a
        *purely unsupervised* setting, where the unsupervised model is not used
        for pre-training. **In this case, the output dimension of the model
        must be specified extra through the** ``model_init_kwargs`` argument.**

        .. warning::

            If the output dimension is given both by the dataset and by the
            kwargs, the **kwargs** take priority. It is assumed that a purely
            unsupervised setting applies. (Datasets may serve multiple purposes
            while the model is already set up for a more specific purpose.)

        If a ``model`` is passed, the output dimension is simply copied from
        the model, disregarding the dataset (since the initialized model will
        be using this model's dimensions anyway) - the output dimension is NOT
        checked against the dataset in this case (while the input, of course,
        is).

        :type data: Dataset
        :param data: The dataset on which the model will be run. Note that this
            setup typically expects that the dataset knows in advance what its
            both input and output dimensions are; in a purely unsupervised
            setting, we'll have to deal with ``n_out`` separately. TODO!!!

        :type model: BaseUnsupervisedModel
        :param model: A model instance that the setup should use.

        :type batch_size: int
        :param batch_size: how many data items will be in one minibatch
            (the data is split to minibatches for training,
            validation and testing)

        :type learning_rate: theano.config.floatX
        :param learning_rate: How fast will the model move in the direction
            of the gradient.

        :type heavy_debug: bool
        :param heavy_debug: Turns on debug prints from Theano functions.

        :type model_init_kwargs: kwargs
        :param model_init_kwargs: Various keyword arguments that get passed
            to the model constructor. See constructor argument documentation.

            .. warning::

                In a purely unsupervised setting, the dataset doesn't define
                the output dimension. In this case, ``n_out`` must be supplied
                as a keyword argument here.

        :rtype: ModelHandle
        :returns: ``ModelHandle(model, train_f, validate_f, test_f)``
            where ``model`` is the Model instance initialized during
            :func:`setup` and the ``_func`` variables are compiled
            theano.functions to use in a learner.

        """
        index = TT.lscalar() # index of minibatch
        X = TT.matrix('X')   # data as a matrix
        X.tag.test_value = numpy.ones((10,data.n_in), dtype=theano.config.floatX)
        # There is no response vector.

        # Check for kwargs ... obsolete?
#       cls._check_init_args(model_init_kwargs)


        # Construct the model instance, or use supplied and do sanity checks.
        if model is None:

            if 'n_out' in model_init_kwargs:
                n_out = model_init_kwargs['n_out']
            elif hasattr(data, 'n_out') and isinstance(data.n_out, int):
                model_init_kwargs['n_out'] = data.n_out
            else:
                raise ValueError('Must supply n_out either from dataset or **model_init_kwargs.')

            # n_out is supplied in model_init_kwargs, either way.
            model = cls(inputs=X, n_in = data.n_in,
                        **model_init_kwargs)
        else:
            # Sanity (dimensionality...) checks:
            # - Are we passing a model of the same type as we're trying
            #   to set up?
            # - Are we passing a dataset that the model can work on?
            assert cls == type(model)
            assert model.n_in == data.n_in

            # Unsupervised model output dimension is not checked against the
            # dataset output dimension.
            # If we are passing an unsupervised model, maybe we are doing so
            # in a purely unsupervised setting, so output size doesn't have
            # to fit the data output size (and the dataset may not have an
            # output size attribute set, anyway).

            model.inputs = X # THIS IS UGLY.
                             # Should re-write setup to
                             # directly use model.input.
                             # (And so should other
                             # methods, so the only thing the setup
                             # really does is link the dataset to
                             # the model INPUTS.)

        # This is a key difference in un-supervision: cost is computed w.r.t.
        # inputs, not a response variable.
        # TODO: REFACTORING NOTE: could this be automatically bound inside
        #       the model's _training_updates method?
        bound_cost = model._cost(model.inputs)

        # Just nudging it a little...
        #updater = safire.learning.learners.ResilientBackpropUpdater(model.params)
        updater = StandardSGDUpdater(learning_rate)

        updates = model._training_updates(updater=updater,
                                          cost=bound_cost)

        # Compile a Theano function that trains the model: returns the cost
        # and updates the model parameters based on the rules defined by the
        # model._training_updates() function.
        training_kwargs = {}
        if heavy_debug:
            training_kwargs['mode'] = theano.compile.MonitorMode(
                        post_func=safire.utils.detect_nan).excluding(
                                            'local_elemwise_fusion', 'inplace')

        dummy_inputs = data.train_X_batch(0, 2).astype(theano.config.floatX)
        model.inputs.tag.test_value = dummy_inputs

        #print 'Dummy inputs: ', model.inputs.tag.test_value

        train_model = theano.function(inputs = [model.inputs],
                                      outputs = bound_cost,
                                      updates = updates,
                                      allow_input_downcast=True,
                                      **training_kwargs)

        # Compile a Theano function that computes the cost that are made
        # by the model on a minibatch of devel/test data
        model.inputs.tag.test_value = dummy_inputs
        validate_model = theano.function(inputs = [model.inputs],
                                outputs = model.error(model.inputs),
                                allow_input_downcast=True,
                                **training_kwargs)

        test_model = theano.function(inputs = [model.inputs],
                                outputs = model.error(model.inputs),
                                allow_input_downcast=True,
                                **training_kwargs)

        run_model = theano.function(inputs = [model.inputs],
                                    outputs = model.outputs,
                                    allow_input_downcast=True)
        

        return ModelHandle(model, train_model, validate_model, test_model,
                           run_model)
