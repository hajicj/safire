#!/usr/bin/env python
"""The SGDLearner module contains learner classes for stochastic gradient
descent.

A **Learner** is a class that gets a model handle and a dataset and optimizes
the model parameters.
"""

import logging
import os
import random
import time
import gensim

import numpy
import theano
import matplotlib.pyplot as plt
#from safire.data.loaders import ModelLoader # Was a circular import
from safire.learning.interfaces.model_handle import BackwardModelHandle
from safire.learning.models import BaseModel

from safire.learning.models.base_supervised_model import BaseSupervisedModel
from safire.learning.models.base_unsupervised_model import BaseUnsupervisedModel
from safire.data.supervised_dataset import SupervisedDataset
import safire.utils


class BaseSGDLearner(gensim.utils.SaveLoad):
    """Base class for other Learner classes. Defines the Learner interface.
    Implements a basic Stochastic Gradient Descent algorithm and provides some
    elementary training environment.

    Using a Learner:

    >>> model_handle = Model.setup()
    >>> dataset = MultimodalShardedDatasetLoader.load()
    >>> learner.evaluate(model_handle, dataset)
    Average cost: 0.897
    >>> learner = BaseSGDLearner(n_epochs=10, **kwargs)
    >>> learner.run(model_handle, dataset)
    >>> learner.evaluate(model_handle, dataset)
    Average cost: 0.017

    The learner can also log and save its progress. This is done using the
    ``track_weights`` and ``track_weights_change`` for logging weights and
    the ``set_saving()`` method for saving intermediate models at each K epochs.

    >>> mloader = ModelLoader('test-data', 'test-data')
    >>> learner.set_saving(mloader, save_every=1, infix='.test_label')
    >>> learner.run(model_handle, dataset)
    >>> os.path.exists(mloader.model_full_path(learner._generate_stage_infix(3)))
    True
    >>> os.path.exists(mloader.model_full_path(learner._generate_stage_infix(11)))
    False

    In the spirit of providing the training environment, learners can also
    try to resume training where it left off by attempting to load a saved
    intermediate file.

    >>> learner.run(model_handle, dataset, resume=True, force_resume=False)
    """

    def __init__(self, n_epochs, b_size, learning_rate=0.13,
                 patience=5000, patience_increase=100,
                 improvement_threshold=0.995, validation_frequency=None,
                 track_weights=False, track_weights_change=False,
                 monitoring=True, shuffle_batches=False,
                 plot_transformation=False, plot_weights=False,
                 plot_every=10, plot_on_init=False):
        """Initialize the learner.

        The parameters common to each learner are just the number of epochs.
        There will be further specializations.

        :type n_epochs: int
        :param n_epochs: How many training epochs should be run?

        :type b_size: int
        :param b_size: What should the size of one training batch be?

        :type learning_rate: float
        :param learning_rate: The SGD learning rate.

        :type patience: int
        :param patience: Always look at at least this many examples.

        :type patience_increase: int
        :param patience_increase: How much patience should increase when a new
            best is found.

        :type track_weights: bool
        :param track_weights: If set to True, will print out a sub-matrix of the
            first layer weights for each batch. The sub-matrix is [0,0],[0,9],
            [9,0] and [9,9].

        :type track_weights_change: bool
        :param track_weights_change: If set to True, will find the 3 largest
            weight changes in each operation.

        :type monitoring: bool
        :param monitoring: Turn on monitoring of training cost, validation cost
            [DEPRECATED: monitoring always on, plot results on demand.]

        :type shuffle_batches: bool
        :param shuffle_batches: If set, will randomly shuffle batch order
            between SGD runs.

        :type plot_transformation: bool
        :param plot_transformation: If set, will plot a sample of the output
            transformation after each epoch and at the end of learning.

        :type plot_weights: bool
        :param plot_weights: If set, will plot the weights in each iteration.
            Only works for single-layer models.

        :type plot_every: int
        :param plot_every: Plot transformations/weights every this many epochs.

        :type plot_on_init: bool
        :param plot_on_init: Plot initial state of model/transformation/recons.
        
        """
        self.n_epochs = n_epochs
        self.b_size = b_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold

        self.validation_frequency = validation_frequency

        self.track_weights = track_weights
        self.track_weights_change = track_weights_change
        self.monitoring = monitoring
        self.shuffle_batches = shuffle_batches

        # Visualizing intermediate results
        self.plot_every = plot_every
        self.plot_transform_per_epoch = plot_transformation
        self.plot_transform = plot_transformation
        #self.plot_every_batch = False
        self.plot_weights = plot_weights
        self.plot_on_init = plot_on_init

        # Saving options.
        self.saving = False
        self.intermediate_infix = None
        self.mloader = None
        self.save_every = None
        self.intermediate_overwrite = None
        self.intermediate_fnames = []
        self.intermediate_tids = []
        self.last_saved_epoch = 0.0

    def training_updates(self, model, cost, learning_rate=0.13,
                         **gradient_kwargs):
        """
        Builds the training function. To implement various optimization
        strategies (rprop, conjugate gradients, levenberg-marquart, etc.),
        override this method.

        :param model:

        :param kwargs:

        :return: Updates.
        """
        X = theano.tensor.dmatrix('X_train')
        model.inputs = X

        gradients = theano.tensor.grad(cost, model.params, **gradient_kwargs)

        updates = []
        for param, gradient in zip(model.params, gradients):
            updates.append((param, param - learning_rate * gradient))

        return updates

    def run(self, model_handle, data, resume=False, force_resume=False):
        """Runs the learner. Returns the model handle.

        Intermediate result saving and progress logging depends on how the
        learner has been set up (see :meth:`set_saving`).

        :type model_handle: ModelHandle
        :param model_handle: A model handle that came out of a ``setup``
            classmethod of a model.

        :type data: Dataset
        :param data: A dataset on which the learner should run.

        :type resume: bool
        :param resume: If given, will attempt to load the last intermediate file
            saved by the learner. (Useful for resuming interrupted training.)

        :type force_resume: bool
        :param force_resume: If set to ``True`` and resuming fails, will raise a
            ``ValueError``. If set to ``False`` and resuming fails, will warn
            and train from scratch.
        """

        # Attempting to resume model training.
        resume_successful = False
        if resume:
            resuming_instance = self.attempt_resume()
            if not resuming_instance:
                if force_resume:
                    raise ValueError('Could not resume training!')
                else:
                    logging.warn('Could not resume training, starting from scratch.')
            else:
                resume_successful = True
                # TODO: There should be a sanity check here!
                # Also, the loading could possibly use TIDs - which would require,
                # however, a ModelLoader.
                model_handle.model_instance = resuming_instance

        try:
            backward_handle = BackwardModelHandle.clone(model_handle)
        except AttributeError:
            backward_handle = None

        # Initialize training
        n_train_batches = data.n_train_batches(self.b_size)
        n_devel_batches = data.n_devel_batches(self.b_size)
        #n_test_batches = data.n_test_batches(self.b_size)

        if not self.validation_frequency:
            self.validation_frequency = min(n_train_batches, self.patience / 2)


        best_params = None
        best_validation_loss = numpy.inf
        best_test_loss = numpy.inf

        done_looping = False
        epoch = 0.0
        max_epoch = self.n_epochs
        if resume and resume_successful:
            epoch = self.last_saved_epoch
            max_epoch += self.last_saved_epoch # Always run for self.n_epochs,
                                               # even if resuming.
        iteration = 0.0

        start_time = time.clock()
        last_epoch_time = start_time

        # Validation init logging
        validation_losses = [ self._validate_batch(model_handle, data, vb_index)
                              for vb_index in xrange(n_devel_batches)]

        validation_loss = numpy.mean(validation_losses)

        logging.info(
            'At initialization: v. error %f' % (validation_loss)
        )

        #if self.track_weights_change:
            #print self.report_weights(model_handle.model_instance)

        self.weights_snapshot(model_handle.model_instance)

        self.monitor = { 'training_cost' : [], 'validation_cost' : [] }
        self.monitor['validation_cost'].append([iteration, validation_loss])

        if self.plot_on_init:
            self.plot_transformed_results(data, model_handle,
                                          backward_handle=backward_handle,
                                          with_orig=True,
                                          title='After init, before training')

        # Run training, use validation
        while not done_looping and epoch < max_epoch:

            batch_ordering = range(n_train_batches)
            if self.shuffle_batches:
                random.shuffle(batch_ordering)

            for b_index in batch_ordering: # One uninterrupted run
                                           # of SGD on train data

                # Runs one iteration of training, based on model and dataset
                # type.
                training_cost = self._train_batch(model_handle, data, b_index)
                self.monitor['training_cost'].append([iteration+1, training_cost])

                #if self.plot_every_batch:
                #    self.plot_transformed_results(data, model_handle)

                if (self.validation_frequency
                    and (iteration + 1) % self.validation_frequency == 0):

                    validation_losses = [ self._validate_batch(model_handle,
                                                               data, vb_index)
                                          #/ float(self.b_size)
                                          for vb_index in xrange(n_devel_batches)]

                    validation_loss = numpy.mean(validation_losses)
                    self.monitor['validation_cost'].append([(iteration+1)/float(self.validation_frequency),
                                                            validation_loss])

                    logging.info(
                        'Epoch %d, minibatch %d/%d: v. error %f' % (
                            epoch, b_index + 1, n_train_batches,
                            validation_loss)
                                 )

                    if self.track_weights_change:
                        print self.report_max_weight_change(model_handle.model_instance)
                        self.weights_snapshot(model_handle.model_instance)

                    if self.track_weights:
                        print self.report_weights(model_handle.model_instance)

                    # Update algorithm based on improved validation score
                    if validation_loss < (best_validation_loss
                                          * self.improvement_threshold):
                        self.patience = max(self.patience,
                                            iteration * self.patience_increase)

                        best_validation_loss = validation_loss

                        # Get corresponding test loss for logging purposes
                        test_loss = self.evaluate(model_handle, data)
                        logging.info('Epoch %i, minibatch %i/%i: t. error %f'%(
                                     epoch, b_index + 1, n_train_batches,
                                     test_loss))

                        if test_loss < best_test_loss:
                            best_test_loss = test_loss

                iteration += 1
                if iteration % 1000 == 0:
                    logging.info('Iteration not. %d' % iteration)

                if self.patience <= iteration:
                    done_looping = True
                    break # from cycle over minibatches, not cycle over epochs

            # Epoch logging
            epoch_time = time.clock()
            logging.info('----------------------------------')
            logging.info('Epoch %i report:' % epoch)
            logging.info('    Complete in %f s. (average %f / epoch)' % (
                epoch_time - last_epoch_time,
                (epoch_time - start_time) / (epoch + 1)))
            logging.info('    Average time per minibatch: %f' % (
                (epoch_time - last_epoch_time) / n_train_batches))
            logging.info('    Best validation loss: %f' % best_validation_loss)
            logging.info('    Best test loss: %f' % best_test_loss)
            logging.info('      Patience left: %d' % self.patience)
            last_epoch_time = epoch_time

            epoch += 1

            if self.saving and epoch % self.save_every == 0:
                # Saves an intermediate model.
                self.save_intermediate(model_handle.model_instance, str(epoch))
                self.last_saved_epoch = epoch

            if self.plot_transform_per_epoch and epoch % self.plot_every == 0:
                self.plot_transformed_results(dataset=data,
                                              model_handle=model_handle,
                                              backward_handle=backward_handle,
                                              plot_bias=True,
                                              title='Test batch transformed after epoch %d' % epoch)

            if self.plot_weights and epoch % self.plot_every == 0:
                W = self._get_weights_from_model(model_handle.model_instance)
                weights = W.get_value(borrow=True)
                safire.utils.heatmap_matrix(weights,
                                    title='Weights heatmap af epoch %d' % epoch,
                                    with_average=False,
                                    colormap='coolwarm',
                                    vmin=0.9 * numpy.min(weights),
                                    vmax=0.9 * numpy.max(weights))

                W_diff = weights - self.W_snapshot
                self.weights_snapshot(model_handle.model_instance)
                safire.utils.heatmap_matrix(W_diff,
                                            title='Diff of W from prev. epoch',
                                            with_average=False,
                                            colormap='coolwarm',
                                            vmin=0.9 * numpy.min(weights),
                                            vmax=0.9 * numpy.max(weights))


        # Total logging
        end_time = time.clock()
        logging.info('===================================')
        logging.info('Training run report:')
        logging.info('    Total epochs: %i' % epoch)
        logging.info('    Total time: %f' % (end_time - start_time))
        logging.info('    Average time per epoch: %f' % (
                        (end_time - start_time) / epoch))
        logging.info('    Average time per minibatch: %f' % (
            (end_time - start_time) / (epoch * n_train_batches)))
        logging.info('    Best validation score: %f' % best_validation_loss)
        logging.info('    Best test loss: %f' % best_test_loss)

        if self.plot_transform:
            self.plot_transformed_results(dataset=data,
                                          model_handle=model_handle,
                                          backward_handle=backward_handle,
                                          title='Test batch after training.',
                                          plot_bias=True)
        if self.plot_weights:
            W = self._get_weights_from_model(model_handle.model_instance)
            weights = W.get_value(borrow=True)
            safire.utils.heatmap_matrix(weights,
                                    title='Weights heatmap af epoch %d' % epoch,
                                    with_average=False,
                                    colormap='coolwarm',
                                    vmin=0.9 * numpy.min(weights),
                                    vmax=0.9 * numpy.max(weights))

            W_diff = weights - self.W_snapshot
            self.weights_snapshot(model_handle.model_instance)
            safire.utils.heatmap_matrix(W_diff,
                                            title='Diff of W from prev. epoch',
                                            with_average=False,
                                            colormap='coolwarm',
                                            vmin=0.9 * numpy.min(weights),
                                            vmax=0.9 * numpy.max(weights))

        return best_validation_loss

    def set_saving(self, model_loader, save_every, infix=None, overwrite=False):
        """
        Settings for saving intermediate models. Currently, will save every K-th
        epoch, using the given infix and a ``.tmp.n`` suffix, for n-th epoch.

        .. note::

          Design-wise, this adds some internal state to the learner and introduces
          a binding from a loader class (data management) inside the training
          architecture. This is not a problem per se - however, it is thus wrong
          to imagine that the loader classes are a *layer* of architecture;
          instead, they are *services* for data management, permeating throughout
          the application. Since loaders have no internal state of their own
          (their "internal state" is the actual state of the data directory),
          this is not a problem; they can be freely shared across the application.

        :type infix: str
        :param infix: The infix to use for the model.

            .. note::

                A temp-string will be appended to this infix for each intermediate
                saved model that includes ``.tmp`` - the best saving infix is
                simply the infix under which you want to save the final model.

        :type model_loader: safire.data.loaders.ModelLoader
        :param model_loader: The model loader that will correctly generate file
            names.

        :type save_every: int
        :param save_every: Each k-th epoch, the learner will save an intermediate
            model.

        :type overwrite: bool
        :param overwrite: If True, will only keep the last intermediate model.

        """
        self.intermediate_infix = infix
        self.mloader = model_loader
        self.save_every = save_every
        self.intermediate_overwrite = overwrite
        self.saving = True

        # Does not reset intermediate files buffer.

    def save_intermediate(self, model, tid):
        """
        Given previously supplied settings, saves the given model. No infix or
        filename is given - it will be generated based on the saving settings.

        :param model: The model to save.

        :param tid: The ID of the stage at which the intermediate model is created.
            If saving at epochs, this will typically be a number. The learner
            will generate this argument during :meth:`run`.

        """
        if not self.saving:
            logging.warn('Saving intermediate model without the \'saving\' '
                         'attribute set; why?')

        temp_infix = self._generate_stage_infix(tid)
        self.mloader.save_model(model, temp_infix)

        # Update last saved name tracking, for overwrite purposes.
        new_name = self.mloader.model_full_path(temp_infix)
        if self.intermediate_overwrite:
            self.clear_intermediate()
        self.intermediate_fnames.append(new_name)
        self.intermediate_tids.append(tid)

    def load_intermediate(self, tid):
        """
        Given a stage ID, can load a model. This is to be able to continue
        training from some model state.

        :param tid:
        :return:
        """
        pass

    def _generate_stage_infix(self, tid):
        """Generates the infix for an intermediate model with infix ``infix``
        saved at stage ``tid``."""
        if self.intermediate_infix is None:
            return '.intermediate.' + str(tid)
        else:
            return self.intermediate_infix + '.intermediate.' + str(tid)

    def clear_intermediate(self):
        """Removes all intermediate files saved so far. Has to keep the list
        of available intermediate files consistent!"""
        processed = []
        processed_tids = []
        n_to_process = len(self.intermediate_fnames) # Keeping track
        try:
            for tid, fname in zip(self.intermediate_tids, self.intermediate_fnames):
                if not os.path.isfile(fname):
                    logging.warn('Somebody already removed a saved intermediate model (%s). Continuing without...' % fname)
                    processed.append(fname)
                    processed_tids.append(tid)
                else:
                    try:
                        os.remove(fname)
                    finally:
                        if not os.path.isfile(fname): # The file has been successflly removed.
                            processed.append(fname)
                            processed_tids.append(tid)
        finally:
            for processed_tid, processed_fname in zip(processed_tids, processed):
                self.intermediate_fnames.remove(processed_fname)
                self.intermediate_tids.remove(processed_tid)
            logging.info('Clearing intermediate files from loader with infix %s: done' % self.intermediate_infix)
            logging.info('    Cleared: %d / %d' % (len(processed), n_to_process))
            logging.info('    Cleared stages: %s' % str(processed_tids))

            self.last_saved_epoch = 0.0

    def attempt_resume(self):
        """Attempts to resume training from last intermediate file saved."""

        # if not hasattr(self, 'mloader'):
        #    raise ValueError("Cannot resume training: model loader is not available.")
        resuming_instance = None
        if len(self.intermediate_fnames) < 1:
            logging.warn("Cannot resume training: no intermediate file tracked.")
            return None

        try:
            model_file = self.intermediate_fnames[-1]
            resuming_instance = BaseModel.load(model_file, load_any=True)
        except Exception as e:
            logging.warn('Could not load resuming model %s' % model_file)
            raise
        else:
            return resuming_instance

    def evaluate(self, model_handle, dataset):
        """Evaluates model performance on the test portion of the given
        dataset.

        :type model_handle: ModelHandle
        :param model_handle: A model handle.

        :type dataset: Dataset
        :param dataset: A dataset that corresponds to the type of the model
            instance in ``model_handle`` - Supervised or Unsupervised.
            A SupervisedDataset is allowed for an UnsupervisedModel, but not
            the other way round.


        """

        test_losses = [ self._test_batch(model_handle, dataset, i)
                        #/ float(self.b_size)
                        for i in xrange(dataset.n_test_batches(self.b_size)) ]

        return numpy.mean(test_losses)

    def _train_batch(self, model_handle, dataset, batch_index):
        """Runs a training batch. It relies on the model handle's
        ``train`` compiled Theano function to expect inputs according
        to whether the model instance is a subclass of
        :class:`BaseSupervisedModel` or :class:`BaseUnsupervisedModel`.

        :type model_handle: ModelHandle
        :param model_handle: A model handle.

        :type dataset: Dataset
        :param dataset: A dataset that corresponds to the type of the model
            instance in ``model_handle`` - Supervised or Unsupervised.
            A SupervisedDataset is allowed for an UnsupervisedModel, but not
            the other way round.

        :type batch_index: int
        :param batch_index: The index of the requested batch.

        """
        model = model_handle.model_instance
        batch_loss = None

        if isinstance(model, BaseSupervisedModel):
            if not isinstance(dataset, SupervisedDataset):
                raise ValueError('Attempting to train supervised model without'+
                                 ' a supervised dataset (dataset type: %s) ' % (
                                     str(type(dataset))))
            train_X = dataset.train_X_batch(batch_index, self.b_size)
            train_y = dataset.train_y_batch(batch_index, self.b_size)

            logging.debug('Train batch: %s' % train_X)

            batch_loss = model_handle.train(train_X, train_y)

        elif isinstance(model, BaseUnsupervisedModel):
            train_X = dataset.train_X_batch(batch_index, self.b_size)

            batch_loss = model_handle.train(train_X)

        return batch_loss

    def _validate_batch(self, model_handle, dataset, batch_index):
        """Runs a validation batch. It relies on the model handle's
        ``validate`` compiled Theano function to expect inputs according
        to whether the model instance is a subclass of
        :class:`BaseSupervisedModel` or :class:`BaseUnsupervisedModel`.

        :type model_handle: ModelHandle
        :param model_handle: A model handle.

        :type dataset: Dataset
        :param dataset: A dataset that corresponds to the type of the model
            instance in ``model_handle`` - Supervised or Unsupervised.
            A SupervisedDataset is allowed for an UnsupervisedModel, but not
            the other way round.

        :type batch_index: int
        :param batch_index: The index of the requested batch.

        """
        model = model_handle.model_instance
        batch_loss = None

        if isinstance(model, BaseSupervisedModel):
            if not isinstance(dataset, SupervisedDataset):
                raise ValueError('Attempting to validate supervised model'+
                        ' without a supervised dataset (dataset type: %s) ' % (
                        str(type(dataset))))

            logging.debug('Batch index type: %s' % type(batch_index))
            logging.debug('Batch size type: %s' % type(self.b_size))

            devel_X = dataset.devel_X_batch(batch_index, self.b_size)
            devel_y = dataset.devel_y_batch(batch_index, self.b_size)

            batch_loss = model_handle.validate(devel_X, devel_y)

        elif isinstance(model, BaseUnsupervisedModel):
            devel_X = dataset.devel_X_batch(batch_index, self.b_size)

            batch_loss = model_handle.validate(devel_X)

        return batch_loss

    def _test_batch(self, model_handle, dataset, batch_index):
        """Runs a test batch. It relies on the model handle's
        ``test`` compiled Theano function to expect inputs according
        to whether the model instance is a subclass of
        :class:`BaseSupervisedModel` or :class:`BaseUnsupervisedModel`.

        :type model_handle: ModelHandle
        :param model_handle: A model handle.

        :type dataset: Dataset
        :param dataset: A dataset that corresponds to the type of the model
            instance in ``model_handle`` - Supervised or Unsupervised.
            A SupervisedDataset is allowed for an UnsupervisedModel, but not
            the other way round.

        :type batch_index: int
        :param batch_index: The index of the requested batch.

        """
        model = model_handle.model_instance
        batch_loss = None

        if isinstance(model, BaseSupervisedModel):
            if not isinstance(dataset, SupervisedDataset):
                raise ValueError('Attempting to validate supervised model'+
                        ' without a supervised dataset (dataset type: %s) ' % (
                        str(type(dataset))))
            train_X = dataset.test_X_batch(batch_index, self.b_size)
            train_y = dataset.test_y_batch(batch_index, self.b_size)

            batch_loss = model_handle.test(train_X, train_y)

        elif isinstance(model, BaseUnsupervisedModel):
            train_X = dataset.test_X_batch(batch_index, self.b_size)

            batch_loss = model_handle.test(train_X)

        return batch_loss

    def plot_transformed_results(self, dataset, model_handle,
                                 title='Dataset heatmap after learner run',
                                 with_orig=False, with_no_bias=False,
                                 plot_bias=False, backward_handle=None):
        """Plots a sample heatmap of how the dataset will be transformed."""
        sample_size = min(1000, (len(dataset) - dataset._test_doc_offset))
        batch = 0 # Deterministic plotting.
        #batch = random.randint(0, dataset.n_test_batches(sample_size))
        sample_data = dataset.test_X_batch(batch, sample_size)
        transformed_data = numpy.array(model_handle.run(sample_data))

        if with_orig:
            safire.utils.heatmap_matrix(sample_data,
                                        title=title + ' - inputs',
                                        with_average=False,
                                        colormap='afmhot',
                                        vmin=0.0, vmax=1.0)
        #if with_no_bias:
        #    forward_weights = model_handle.model_instance.W.get_value(borrow=True)
        #    linear_activation = numpy.dot(sample_data, forward_weights)
        #    safire.utils.heatmap_matrix(linear_activation,
        #                                 title=title + ' - linear act.',
        #                                 with_average=False,
        #                                 colormap='afmhot',
        #                                 vmin=min(linear_activation),
        #                                 vmax=max(linear_activation))


        # safire.utils.heatmap_matrix(transformed_data,
        #                             title=title + '(abs. 0-1 scale)',
        #                             with_average=True,
        #                             colormap='afmhot',
        #                             vmin=0.0, vmax=1.0)
        safire.utils.heatmap_matrix(transformed_data,
                                    title=title + '(rel. min-max scale)',
                                    with_average=True,
                                    colormap='afmhot',
                                    vmin=numpy.min(transformed_data),
                                    vmax=numpy.max(transformed_data))

        # plot the reconstruction
        if backward_handle:
            reconstructed_data = backward_handle.run(transformed_data)
            # safire.utils.heatmap_matrix(reconstructed_data,
            #                             title=title + 'REC. (abs.)',
            #                             with_average=True,
            #                             colormap='afmhot',
            #                             vmin=0.0, vmax=1.0)
            safire.utils.heatmap_matrix(reconstructed_data,
                                        title=title + 'REC. (rel.)',
                                        with_average=False,
                                        colormap='afmhot',
                                        vmin=numpy.min(reconstructed_data),
                                        vmax=numpy.max(reconstructed_data))

        if plot_bias:
            if hasattr(model_handle.model_instance, 'b'):
                bias_hidden = model_handle.model_instance.b.get_value(borrow=True)
            elif hasattr(model_handle.model_instance, 'b_hidden'):
                bias_hidden = model_handle.model_instance.b_hidden.get_value(borrow=True)
            else:
                bias_hidden = None

            if hasattr(model_handle.model_instance, 'b_prime'):
                bias_visible = model_handle.model_instance.b_prime.get_value(borrow=True)
            elif hasattr(model_handle.model_instance, 'b_visible'):
                bias_visible = model_handle.model_instance.b_visible.get_value(borrow=True)
            else:
                bias_visible = None

            if bias_hidden is not None and bias_visible is not None:
                safire.utils.heatmap_matrix(numpy.atleast_2d(bias_hidden),
                                            title='Hidden units bias',
                                            vmin=-abs(numpy.max(bias_hidden)),
                                            vmax=abs(numpy.max(bias_hidden)),
                                            colormap='coolwarm')

            if bias_visible is not None:
                safire.utils.heatmap_matrix(numpy.atleast_2d(bias_visible),
                                            title='Visible units bias',
                                            vmin=numpy.min(bias_visible),
                                            vmax=numpy.max(bias_visible),
                                            colormap='coolwarm')

        # Compute correlation between total activation and item sum

        sample_mean = numpy.mean(sample_data)
        sample_sqdevs = (sample_data - sample_mean) ** 2
        sample_sqdev = numpy.sum(sample_sqdevs)
        sample_mean_sqdev = numpy.mean(sample_sqdevs)
        print 'Sample mean: %.8f, total sq. dev. %.8f, mean sq. dev %.8f' % (
            sample_mean, sample_sqdev, sample_mean_sqdev
        )

        tr_mean = numpy.mean(transformed_data)
        tr_sqdevs = (transformed_data - tr_mean) ** 2
        tr_sqdev = numpy.sum(tr_sqdevs)
        tr_mean_sqdev = numpy.mean(tr_sqdevs)
        print 'Transformed mean: %.8f, total sq. dev. %.8f, mean sq. dev %.8f' % (
            tr_mean, tr_sqdev, tr_mean_sqdev
        )

        # Compute correlation of


    def report_weights(self, model, submatrix=[[[0,0],[0,9]],[[9,0],[9,9]]]):
        """Reports the weight submatrix from the first layer of the model."""
        W = self._get_weights_from_model(model)

        W_value = W.get_value(borrow=True)
        submatrix = [[W_value[i,j] for i,j in srow] for srow in submatrix]

        return submatrix

    def weights_snapshot(self, model):
        """Takes a snapshot of the model weights and stores it to a cache."""
        W = self._get_weights_from_model(model)

        self.W_snapshot = W.get_value()

    def weight_change_vs_snapshot(self, model):
        """Returns the difference in weights between the model and the
        snapshot."""
        W = self._get_weights_from_model(model)

        W_value = W.get_value()

        W_diff = W_value - self.W_snapshot

        return W_diff

    def find_maximum_diff(self, W_diff, k = 3):
        """Finds the k highest differences (in absolute values) in the weight
        difference report."""

        max_diff_w_indices = safire.utils.n_max(numpy.absolute(W_diff), k)

        return max_diff_w_indices

    def report_maximum_diff(self, max_diff_w_indices, W_diff, W):
        """Formats the maximum differences."""
        lines = ['--- diff report ---']
        for diff, idx in max_diff_w_indices:
            lines.append(str(idx) + ' : ' + str(W_diff[idx[0], idx[1]]) + ' --> '
            + str(W[idx[0], idx[1]]))

        return '\n'.join(lines)

    def report_max_weight_change(self, model, k=3):

        W = self._get_weights_from_model(model)
        W_diff = self.weight_change_vs_snapshot(model)
        total_diff = numpy.sum(W_diff)
        max_diff_w_indices = self.find_maximum_diff(W_diff, k)
        report = self.report_maximum_diff(max_diff_w_indices, W_diff, W.get_value(borrow=True))

        if W_diff[0,0] >= 0.0001:
            logging.info('Changing weight of [0,0] by %.5f' % W_diff[0,0])
        logging.info('Weight tracking, total diff: %.5f' % total_diff)
        return report

    def _get_weights_from_model(self, model):
        W = None
        if hasattr(model, 'W'):
            W = model.W
        elif hasattr(model, 'hidden_layers'):
            W = model.hidden_layers[0].W
        elif hasattr(model, 'log_regression_layer'):
            W = model.log_regression_layer.W
        else:
            logging.warn('Cannot report weights - couldn\'t find weight matrix.')

        return W
