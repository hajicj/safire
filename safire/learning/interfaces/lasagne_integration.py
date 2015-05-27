"""
This module contains classes and functions that allow Lasagne models to be
used inside Safire pipelines. Because the ModelHandle class is general enough,
we only need to provide the appropriate theano functions, for example like the
``lasagne.examples.mnist.py:create_iter_functions`` function does. We don't
need to implement a specific handle or transformer, we need to implement the
correct setup() function.

Note that this module should *not* rely on being able to import Lasagne -- these
integration classes should be optional ("if you have Lasagne, you can use it,
if you don't, you don't have to worry about this class").
"""
import logging
import time
import theano
import theano.tensor as TT

import safire
from safire.learning.interfaces.model_handle import LasagneModelHandle

__author__ = "Jan Hajic jr."


class LasagneSetup(object):
    """This class is responsible for taking a Lasagne model and creating
    the appropriate handles for training/validating/testing/running.
    """

    @classmethod
    def setup(self, model,
              loss_expr, loss_inputs, updater, updater_kwargs=dict(),
              monitor_expr=None, monitor_inputs=None,
              run_expr=None, run_inputs=None,
              # X_tensor_type=TT.matrix,
              # y_tensor_type=TT.matrix,
              fn_kwargs=dict(),
              heavy_debug=False):
        """The :meth:`setup` method is where Lasagne is interfaced to Safire:
        it is responsible for producing for a given Lasagne model the set of
        model handles that can train, validate, test and run the model.

        It's up to you to define the loss function for training and the
        monitoring function for validation/evaluation. You also have to supply
        the symbolic variables which should serve as the inputs for those two
        functions, because they need to be passed to the compiled theano
        functions as inputs.

        There are several differences between the Safire model setup() and this
        setup(), however. For example, this setup does *not* need to have the
        dataset available, as the model is *not* being initialized here
        and so the input/output dimensions must have already been set elsewhere.
        Additionally, all Lasagne examples feed data to the networks using
        the ``givens`` mechanism of ``theano.function``. This does not fly with
        datasets larger than memory (or GPU memory), so we feed the data in the
        pylearn2 way through theano.function's ``inputs`` mechanism.

        Adapting the ``mnist.py`` example script from the Lasagne library would
        split the create_iter_function into two parts. First, the loss
        and other output expressions are defined; then, this ``setup()`` method
        is called to produce the compiled functions and wrap them into handles.

        .. code-block:: python

            # We first produce the lasagne-specific code.

            X_batch = X_tensor_type('x')
            y_batch = T.ivector('y')

            # Note that the categorical_crossentropy function is polymorphic,
            # it can deal with y_batch being just category labels and handle
            # them as one-hot vectors internally.
            objective = lasagne.objectives.Objective(output_layer,
                loss_function=lasagne.objectives.categorical_crossentropy)

            loss_inputs = [X_batch, y_batch]
            loss_expr = objective.get_loss(X_batch, target=y_batch)

            # The way updates of model parameters are computed defines
            # the learning algorithm (in this case, nesterov accelerated
            # gradient with momentum).
            updater = lasagne.updates.nesterov_momentum
            updater_settings = {'learning_rate': learning_rate,
                                'momentum': momentum}

            monitor_inputs = [X_batch, y_batch]

            # The mnist.py example monitors both the loss and classification
            # accuracy on the heldout data.
            monitor_cost = objective.get_loss(X_batch, target=y_batch,
                                           deterministic=True)
            pred = T.argmax(
                lasagne.layers.get_output(output_layer, X_batch,
                    deterministic=True), axis=1)
            accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)
            monitor_expr = [monitor_cost, accuracy]

            # Here, we are relying on the default behavior for generating
            # run_expr and run_inputs from the model and monitor_inputs.
            setup_handles = LasagneSetup.setup(output_layer,
                                               updater, updater_settings,
                                               loss_expr, loss_inputs,
                                               monitor_expr, monitor_inputs)

            # Now, we can initialize the learner and transformer.

        Note that Lasagne so far handles only supervised networks.

        :param model: A Lasagne model (output layer, since Lasagne models are
            recursive).

        :param loss_expr: The loss function for training the model. A theano
            expression that takes a dataset batch as input (and a corresponding
            batch as output).

        :param loss_inputs: The symbolic variables that serve as the inputs
            for the loss function.

        :param updater: A ``lasagne.updates`` function that generates
            the symoblic expressions for training updates for the model.

        :param updater_kwargs: Some keyword arguments for the updater (stuff
            like learning rate, momentum, etc.)

        :param monitor_expr: The theano expression that should be used
            as the validation and evaluation output. Defaults to ``loss_expr``.

        :param monitor_inputs: The symbolic variables that are the inputs of
            the monitor expression. Defaults to ``loss_inputs``.

        :param run_expr: The theano expression that generates the network's
            output when supplied with the network's input. (Usually the function
            that returns the activations on the last layer.) If not supplied,
            defaults to lasagne.helper function
            ``get_output(model, run_inputs, deterministic=True)``.

        :param run_inputs: The symbolic variables that are the inputs of
            the run expression. If not set, defaults to ``monitor_inputs[0]``
            (assumes that monitoring is done by measuring performance on some
            heldout data, so that the monitor expression's inputs are actually
            (X, y) pairs of data items and their target values).
            Note that this assumption may be sometimes off.

        :param fn_kwargs: Keyword arguments for the compiled functions.

        :param heavy_debug: Turn on massive logging of theano function innards.

        :rtype: dict(str => ModelHandle)
        :returns: A dict of model handles for training, validating, testing
            and plain old running the model. Pass these to a Learner.

        """
        # X_batch = X_tensor_type('X')
        # y_batch = y_tensor_type('y')

        import lasagne.layers

        params = lasagne.layers.get_all_params(model)
        updates = updater(loss_expr, params, **updater_kwargs)

        if heavy_debug:
            fn_kwargs['mode'] = theano.compile.MonitorMode(
                        post_func=safire.utils.detect_nan).excluding(
                                            'local_elemwise_fusion', 'inplace')

        print 'LasagneSetup.setup(): Compiling train_fn'
        _fn_compile_clock = time.clock()
        train_fn = theano.function(
            loss_inputs, loss_expr,
            updates=updates,
            allow_input_downcast=True,
            **fn_kwargs
        )
        print 'Finished in: {0:.2f} s'.format(time.clock() - _fn_compile_clock)

        if monitor_expr is None:
            monitor_expr = loss_expr
        if monitor_inputs is None:
            monitor_inputs = loss_inputs

        print 'LasagneSetup.setup(): Compiling monitor_fn'
        _fn_compile_clock = time.clock()
        monitor_fn = theano.function(
            monitor_inputs, monitor_expr,
            allow_input_downcast=True,
            **fn_kwargs
        )
        print 'Finished in: {0:.2f} s'.format(time.clock() - _fn_compile_clock)

        validate_fn = monitor_fn
        test_fn = monitor_fn

        # There is some trouble with lasagne.helpers.get_output and
        # theano.function's inputs: the former does *not* take a list while
        # the latter *needs* a list.
        if run_expr is None:
            if run_inputs is None:
                run_inputs = monitor_inputs[0]
                run_expr = lasagne.layers.helper.get_output(model,
                                                            run_inputs,
                                                            deterministic=True)
            else:
                run_expr = lasagne.layers.helper.get_output(model,
                                                            run_inputs,
                                                            deterministic=True)
        elif run_inputs is None:
            run_inputs = monitor_inputs[0]

        if not isinstance(run_inputs, list):
            _run_inputs = [run_inputs]
        else:
            _run_inputs = run_inputs

        print 'LasagneSetup.setup(): Compiling run_fn'
        _fn_compile_clock = time.clock()
        run_fn = theano.function(
            _run_inputs, run_expr,
            allow_input_downcast=True,
            **fn_kwargs
        )
        print 'Finished in: {0:.2f} s'.format(time.clock() - _fn_compile_clock)

        print 'LasagneSetup.setup(): assembling setup_handles dict'
        train_handle = LasagneModelHandle(model, train_fn)
        validate_handle = LasagneModelHandle(model, validate_fn)
        test_handle = LasagneModelHandle(model, test_fn)
        run_handle = LasagneModelHandle(model, run_fn)

        handle_dict = {'train': train_handle,
                       'validate': validate_handle,
                       'test': test_handle,
                       'run': run_handle}

        return handle_dict