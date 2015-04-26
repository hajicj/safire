"""
.. module:: 
    :platform: Unix
    :synopsis: ???

.. moduleauthor: Jan Hajic <hajicj@gmail.com>
"""
import logging
import numpy
import pdb
import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams
import safire

from safire.learning.models.base_unsupervised_model import BaseUnsupervisedModel
from safire.learning.interfaces.pretraining_model_handle import PretrainingModelHandle
from safire.learning.interfaces.model_handle import ModelHandle
from safire.utils import check_kwargs


class RestrictedBoltzmannMachine(BaseUnsupervisedModel):
    """This class implements the Restricted Boltzmann Machine model.

    The RBM-derived models support *sampling*, both forward and backward.
    This is accompolished using the following methods:

    * :meth:`sample_h_given_v()` for forward sampling,

    * :meth:`sample_v_given_h()` for backward sampling,

    * :meth:`sample_vhv()` for getting an input sample for the given input,

    * :meth:`sample_hvh()` for getting a hidden sample for the given hidden
        state,

    * :meth:`mean_h_given_v()` for forward expectation,

    * :meth:`mean_v_given_h()` for backward expectation.

    .. note::

      Methods ``sample_vhv()`` and ``sample_hvh()`` implement sampling both ways:
      both the non-input layer (hidden for ``vhv``, visible for ``hvh``) and then
      the input layer is sampled. If you wish to combine means and samples, use
      a composition of the ``sample_`` and ``mean_`` methods::

         v_mean_from_h_sample = model.mean_v_given_h(model.sample_h_given_v(X))

    However, for computing gradients and such, the naive implementation of these
    methods has problems with numerical stability. Theano can perform
    optimizations to remedy the problem, but in order to do that, it has to
    have access to some intermediate steps. Therefore, this suite of sampling
    methods has a protected counterpart: ``_sample_h_given_v()``, etc., which
    should NOT be called from the outside, but is used in building the
    expression graph for training with (P)CD.
    
    .. warning::
    
        There's some downwright ugly hacks here that have to do with how
        Theano works. Specifically, the training updates cost definition
        have to be in the same method. Other way round: the _cost() cannot
        be computed without the gradient descent updates.

        This, unfortunately, currently does affect how ``setup()`` needs to be
        written. So, RBMs have their own setup.

    """
    def __init__(self, inputs, n_in, n_out=100, 
                 activation=TT.nnet.sigmoid,
                 backward_activation=TT.nnet.sigmoid,
                 W=None, b=None, b_hidden=None, b_visible=None,
                 persistent=None, CD_k=1, CD_use_mean=True,
                 sparsity_target=None, output_sparsity_target=None,
                 numpy_rng=numpy.random.RandomState(),
                 L1_norm=0.0, L2_norm=0.0, bias_decay = 0.0,
                 entropy_loss = 0.0, centering=False, prefer_extremes=None,
                 noisy_input=None, theano_rng=None):
        """ Initialize the parameters of the logistic regression
        A Logistic Regression layer is the end layer in classification
        network.

        :type inputs: theano.tensor.var.TensorVariable
        :param inputs: Symbolic variable that describes the input
                       of the architecture (e.g., one minibatch of
                       input images, or output of a previous layer)


        :type n_in: int
        :param n_in: Number of input units, the dimension of the space
                     in which the data points live

        :type n_out: int
        :param n_out: The number of hidden units.
        
        :type activation: theano.tensor.elemwise.Elemwise
        :param activation: The nonlinearity applied at visible neuron
                           output.

        :type backward_activation: theano.tensor.elemwise.Elemwise
        :param backward_activation: The nonlinearity applied at hidden neuron
                           output. If not given, same as ``activation``. (Some
                           RBMs, like the Replicated Softmax model, use a
                           different forward and backward activation function.)

        :type W: theano.tensor.sharedvar.TensorSharedVariable
        :param W: Theano variable pointing to a set of weights that should
                  be shared between the autoencoder and another architecture;
                  if autoencoder should be standalone, leave this as None.
                  This set of weights refers to the transition from visible
                  to hidden layer.
        
                        
        :type b: theano.tensor.sharedvar.TensorSharedVariable
        :param b: Theano variable pointing to a set of bias values that
                  should be shared between the autoencoder and another
                  architecture; if autoencoder should be standalone,
                  leave this as None. This set of bias values refers
                  to the transition from visible to hidden layer. 

                  .. note:

                    The ``b`` name is used in the RBM for compatibility
                    of class interface. Internally, the name ``b_hidden``
                    is used to improve clarity of the sometimes more
                    complicated math expressions, and for ontological
                    symmetry with ``b_visible``.

        :type b_hidden: theano.tensor.sharedvar.TensorSharedVariable
        :param b: Alias for b, used internally as the attribute name
            to make the purpose clear.
                  
            .. warn:

              Do not use both ``b`` and ``b_hidden`` at the same time!
              The intended interface is ``b``, which is also used in
              the ``link()`` class method to construct the RBM.

                  
        :type b_visible: theano.tensor.sharedvar.TensorSharedVariable
        :param b_visible: Theano variable pointing to a set of bias values
                        that should be shared between the autoencoder and
                        another architecture; if autoencoder should be 
                        standalone, leave this as None. This set of bias
                        values refers to the transition from visible to 
                        hidden layer. 
                        
        :type persistent: theano.tensor.sharedvar.TensorSharedVariable
        :param persistent: If you wish to train using Persistent Contrastive
            Divergence, supply an initial state of the Markov chain. If set to
            None (default), use Contrastive Divergence for training
            (initialize the chain to the current data point).

        :type CD_k: int
        :param CD_k: How many Gibbs sampling steps should Contrastive
            Divergence take in generating the negative particle.
            
        :type CD_use_mean: Boolean
        :param CD_use_mean: Should the (P)CD Gibbs chain end use the mean
            activation of the visible units as the chain end? If ``False``,
            uses the visible sample. If ``True``, uses the visible mean.
            Default is ``True``.

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

        :type entropy_loss: float
        :param entropy_loss: If nonzero, will add to cost the entropy of each
            neuron.

        :type prefer_extremes: float
        :param prefer_extremes: Adds a - sum(log((2X - 1) ** 2)) with this
            coefficient.

        :type centering: bool
        :param centering: If given, will use the centering trick.

        :type noisy_input: float
        :param noisy_input: If given, will add noise to the inputs uniformly
            sampled from ``(0, noisy_input)``.
        """
        logging.debug('Initializing RBM model - entered constructor.')

        logging.debug('Initializing superclass.')
        super(RestrictedBoltzmannMachine, self).__init__(inputs, n_in, n_out)

        self.activation = activation
        if not backward_activation:
            backward_activation = activation
        self.backward_activation = backward_activation

        logging.debug('Initializing weights.')
        if not W:
            W = self._init_weights('W', (n_in, n_out), numpy_rng)
        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(W, (n_in, n_out))

        self.W = W

        logging.debug('Initializing hidden bias.')
        if b and b_hidden:
            raise TypeError('Cannot call constructor with both \'b\' and \
                            \'b_hidden\' supplied.')

        # Mask b as b_hidden for the purposes of the class.
        if b:
            b_hidden = b

        if not b_hidden:
            b_hidden = self._init_bias('b_hidden', n_out, numpy_rng)
            # initialize the biases b_hidden as a vector of n_out 0s
            #b_hidden = theano.shared(value = numpy.zeros((n_out,),
            #                      dtype = theano.config.floatX),
            #                  name = 'b_hidden')

        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(b_hidden, (n_out,))

        self.b_hidden = b_hidden

        logging.debug('Initializing visible bias.')
        if not b_visible:
            # initialize the biases b_hidden as a vector of n_out 0s
            b_visible = self._init_bias('b_visible', n_in, numpy_rng)
        else:    # Check for consistency in supplied weights
            self._init_param_consistency_check(b_visible, (n_in,))

        self.b_visible = b_visible
        
        # Different params for tied weights!
        # This will be difficult to put in a general method.
        self.params = [self.W, self.b_hidden, self.b_visible]

        logging.debug('Initializing RBM-specific parameters.')
        # RBM-specific parameters
        self.persistent = persistent
        self.CD_k = CD_k
        self.CD_use_mean = CD_use_mean

        self.sparsity_target = sparsity_target
        self.output_sparsity_target = output_sparsity_target
        self.L1_norm = L1_norm
        self.L2_norm = L2_norm
        self.bias_decay = bias_decay
        self.prefer_extremes = prefer_extremes
        self.noisy_input = noisy_input

        self.centering = centering
        self.centering_offset = None
        if self.centering:
            self.centering_offset = theano.shared()
        
        self._cost_and_updates_computed = False
        self.__rng_update = None # This is a Theano technicality: will need it
                                 # to keep _cost() and _training_updates()
                                 # interfaces consistent, but due to stuff with
                                 # theano.scan() and a Theano rng that has to
                                 # export its update from the _cost() method
                                 # to the _training_updates() method
        
        if theano_rng is None:
          theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        self.theano_rng = theano_rng
        self.numpy_rng = numpy_rng

        logging.debug('Initializing outputs.')
        self.outputs = self.activation(TT.dot(self.inputs, self.W) + self.b_hidden)

        logging.debug('RBM constructor done.')


    def _init_args_snapshot(self):
        """Saves the model in the form of an init kwarg dict, since not all
        attributes of the instance can be pickled. Upon loading, the saved
        model kwarg dict will be used as ``**kwargs`` (the ``load`` method
        is a classmethod) for an initialization of the model."""

        init_arg_dict = {
            'W' : self.W,
            'b_hidden' : self.b_hidden,
            'b_visible' : self.b_visible,
            'n_in' : self.n_in,
            'n_out' : self.n_out,
            'activation' : self.activation,
            'backward_activation' : self.backward_activation,
            'inputs' : self.inputs,
            'persistent' : self.persistent,
            'CD_k' : self.CD_k,
            'CD_use_mean' : self.CD_use_mean,
            'sparsity_target' : self.sparsity_target,
            'output_sparsity_target' : self.output_sparsity_target,
            'L1_norm' : self.L1_norm,
            'L2_norm' : self.L2_norm,
            'bias_decay' : self.bias_decay,
            'centering' : self.centering,
            'prefer_extremes' : self.prefer_extremes
        }

        return init_arg_dict

    def mean_h_given_v(self, visible):
        """Computes expected activation on hidden units."""
        if self.noisy_input:
            new_visible = safire.utils.matutils.uniform_noise(visible,
                                                              self.noisy_input,
                                                              self.theano_rng)
        else:
            new_visible = visible
        return self.activation(TT.dot(new_visible, self.W) + self.b_hidden)

    def sample_h_given_v(self, visible):
        """Samples the hidden layer given the visible layer.

        As opposed to the protected ``_sample_h_given_v()`` method, only returns
        the sample."""
        mean_h = self.activation(TT.dot(visible, self.W) + self.b_hidden)
        #pdb.set_trace()
        sample_h = self.theano_rng.binomial(size=mean_h.shape,
                                            n=1, p=mean_h,
                                            dtype=theano.config.floatX)
        return sample_h

    def sample_vhv(self, visible):
        """Performs one Gibbs sampling step from visible to visible layer."""
        return self.sample_v_given_h(self.sample_h_given_v(visible))

    def mean_v_given_h(self, hidden):
        """Computes the expected visible layer activation the hidden layer."""
        return self.backward_activation(TT.dot(hidden, self.W.T) + self.b_visible)

    def sample_v_given_h(self, hidden):
        """Samples the visible layer given the hidden layer."""
        mean_v = self.backward_activation(TT.dot(hidden, self.W.T) + self.b_visible)
        sample_v = self.theano_rng.binomial(size=mean_v.shape,
                                            n=1, p=mean_v,
                                            dtype=theano.config.floatX)
        return sample_v

    def sample_hvh(self, hidden):
        """Performs one Gibbs sampling step from hidden to hidden layer."""
        return self.sample_h_given_v(self.sample_v_given_h(hidden))

    ##########################################################################
    # Functions that implement sampling one layer given the other internally #
    ##########################################################################
    
    def _mean_h_given_v(self, visible):
        """Computes the hidden layer given a visible layer. Returns both
        the hidden layer mean AND the pre-activation TT.dot(...) form, for
        obscure Theano reasons (numerical stability optimizations further
        down the line). I might figure some way around this later.
        
        :type visible: theano.tensor.var.TensorVariable
        :param visible: The state of the visible units (either 1/0, or mean -
            not important).
            
        :rtype: list[theano.function.type(???), theano.tensor.TensorType]
        :returns: A list where the first member is the pre-nonlinearity
            activation of the hidden layer and the second is the mean 
            activation of hidden units after the nonlinearity is applied.
        """    
        pre_activation_mean_h = TT.dot(visible, self.W) + self.b_hidden
        mean_h = self.activation(pre_activation_mean_h)
        return [pre_activation_mean_h, mean_h]
    
    def _mean_v_given_h(self, hidden):
        """Computes the hidden layer given a visible layer. Returns both
        the hidden layer mean AND the pre-activation TT.dot(...) form, for
        obscure Theano reasons (numerical stability optimizations further
        down the line). I might figure some way around this later.
        
        :type hidden: theano.tensor.var.TensorVariable
        :param hidden: The state of the hidden units (either 1/0, or mean -
            not important).
            
        :rtype: list[theano.function.type(???), theano.tensor.TensorType]
        :returns: A list where the first member is the pre-nonlinearity
            activation of the visible layer and the second is the mean 
            activation of visible units after the nonlinearity is applied.
        """    
        pre_activation_mean_v = TT.dot(hidden, self.W.T) + self.b_visible
        mean_v = self.backward_activation(pre_activation_mean_v)
        return [pre_activation_mean_v, mean_v]
    
    def _sample_h_given_v(self, visible):
        """Samples the hidden layer given the activation of the visible layer
        (mean or sample, doesn't matter).
        
        :type visible: theano.tensor.var.TensorVariable
        :param visible: The state of the visible units (either 1/0, or mean -
            not important).
            
        :rtype: list[theano.function.type(???), theano.tensor.TensorType,
            theano.tensor.TensorType]
        :returns: A list where the first member is the pre-nonlinearity
            activation of the hidden layer, the second is the mean 
            activation of hidden units after the nonlinearity is applied
            and the third is the hidden layer sample from the given mean.
        """
        [pre_activation_mean_h, mean_h] = self._mean_h_given_v(visible)
        sample_h = self.theano_rng.binomial(size = mean_h.shape, 
                                            n=1, p = mean_h, 
                                            dtype=theano.config.floatX)
        return [pre_activation_mean_h, mean_h, sample_h]
        
    def _sample_v_given_h(self, hidden):
        """Samples the visible layer given the activation of the hidden layer
        (mean or sample, doesn't matter).
        
        :type hidden: theano.tensor.var.TensorVariable
        :param hidden: The state of the hidden units (either 1/0, or mean -
            not important).
            
        :rtype: list[theano.tensor.var.TensorVariable, 
            theano.tensor.var.TensorVariable,
            theano.tensor.var.TensorVariable]
        :returns: A list where the first member is the pre-nonlinearity
            activation of the visible layer, the second is the mean 
            activation of visible units after the nonlinearity is applied
            and the third is the visible layer sample from the given mean.
        """
        [pre_activation_mean_v, mean_v] = self._mean_v_given_h(hidden)
        sample_v = self.theano_rng.binomial(size = mean_v.shape, 
                                            n=1, p = mean_v, 
                                            dtype=theano.config.floatX)
        return [pre_activation_mean_v, mean_v, sample_v]
    
    def gibbs_step_vhv(self, visible):
        """Given the activation of the visible units (mean or sample, doesn't
        matter), perform one step of Gibbs sampling and return all partial results.
        
        Re-samples the visible layer from the hidden layer *sample*, not mean.
               
        :type visible: theano.tensor.var.TensorVariable
        :param visible: The state of the visible units (either 1/0, or mean -
            not important).
            
        :rtype: list[theano.tensor.var.TensorVariable x 6]
        :returns: A list of Gibbs sampling results and partial resuls. See
            documentation on :func:`mean_h_given_v` for why pre-nonlinearity
            activation is also returned. The returned values are::
            
                [pre_activation_mean_h, mean_h, sample_h, 
                pre_activation_mean_v, mean_v, sample_v]
                
            (See documentation of :func:`sample_v_given_h` and 
            :func:`sample_h_given_v` on what they are.)
        """
        pre_activation_mean_h, mean_h, sample_h = self._sample_h_given_v(visible)
        pre_activation_mean_v, mean_v, sample_v = self._sample_v_given_h(sample_h)
        return [pre_activation_mean_h, mean_h, sample_h,
                pre_activation_mean_v, mean_v, sample_v]

    def gibbs_step_hvh(self, hidden):
        """Given the activation of the hidden units (mean or sample, doesn't
        matter), perform one step of Gibbs sampling and return all partial results.
        
        Re-samples the hidden layer from the visible layer *sample*, not mean.
               
        :type hidden: theano.tensor.var.TensorVariable
        :param hidden: The state of the hidden units (either 1/0, or mean -
            not important).
            
        :rtype: list[theano.tensor.var.TensorVariable x 6]
        :returns: A list of Gibbs sampling results and partial resuls. See
            documentation on :func:`mean_h_given_v` for why pre-nonlinearity
            activation is also returned. The returned values are::
            
                [pre_activation_mean_v, mean_v, sample_v, 
                pre_activation_mean_h, mean_h, sample_h]
                
            (See documentation of :func:`sample_v_given_h` and 
            :func:`sample_h_given_v` on what they are.)
        """

        pre_activation_mean_v, mean_v, sample_v = self._sample_v_given_h(hidden)
        pre_activation_mean_h, mean_h, sample_h = self._sample_h_given_v(sample_v)
        return [pre_activation_mean_v, mean_v, sample_v,
                pre_activation_mean_h, mean_h, sample_h]

        
    def free_energy(self, visible):
        """Computes the free energy of the model.

        :type visible: theano.tensor.TensorType
        :param visible: The state of the visible units (either 1/0, or mean -
            not important).
        
        :rtype: theano.tensor.var.TensorVariable
        :returns: The free energy of the model, given the visible activation.
            Computed as 
            
            .. math::
               :label: free_energy

                \mathcal{F}(x) = - \log \sum_h e^{-E(x,h)}
        """

        pre_activation_mean_h = TT.dot(visible, self.W) + self.b_hidden
        # I don't really know what this does:
        hidden_term = TT.sum(TT.log(1 + TT.exp(pre_activation_mean_h)), axis=1)
        b_visible_term = TT.dot(visible, self.b_visible)
        free_energy = - hidden_term - b_visible_term
        return free_energy
    
    def _cost_and_training_updates(self, X, **kwargs):
        """Returns the symbolic expression for the cost function of
        the network used for training and the corresponding updates.
        
        .. warn:
          
          This method is intended to be only run ONCE per model life.
          No guarantees on what happens if you run it a second time.
        
        We're using the expression

        .. math::
          :label: free_energy_grad

          - \frac{\partial  \log p(x)}{\partial \theta}
           &= \frac{\partial \mathcal{F}(x)}{\partial \theta} -
                 \sum_{\tilde{x}} p(\tilde{x}) \
                     \frac{\partial \mathcal{F}(\tilde{x})}{\partial \theta}.
        
        for the cost, since it's the form the gradient of the log probability
        takes.
        
        :type X: theano.tensor.var.TensorVariable
        :param X: The input data on which to compute cost. (Expected shape:
            batch_size * self.n_in) Maps to 'visible'.
        """
        # Compute positive phase (the 'p' in ph_)
        pre_sigmoid_ph, ph_mean, ph_sample = self._sample_h_given_v(X)
        
        # Compute negative phase (the 'n' in 'nv_', 'nh_')
        # chain_start is the starting 
        if self.persistent is None:
            chain_start = ph_sample
        else:
            chain_start = self.persistent
            
        # Note that the 'updates' variable gets initialized
        # here. It contains the theano_rng update expression
        # that is used during theano.scan() in sampling the
        # hidden/visible units from their means.
        [pre_sigmoid_nvs, nv_means, nv_samples, 
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_step_hvh, 
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps = self.CD_k)
        
        chain_end = nv_samples[-1]
        if self.CD_use_mean:
            chain_end = nv_means[-1]
        
        cost = TT.mean(self.free_energy(X)) - TT.mean(self.free_energy(chain_end))

        # Various means of adjusting the cost
        if self.sparsity_target:
            cost += self._sparsity_cross_entropy(X)

        if self.output_sparsity_target:
            cost += self._output_sparsity_cross_entropy(X)

        # extremes preference function: -log((2*X - 1.0)**2)
        if self.prefer_extremes:
            mean_act = self.activation(TT.dot(X, self.W) + self.b_hidden)
            cost +=  self.prefer_extremes * TT.mean(-TT.log((2.0 * mean_act - 1.0) ** 2 + 0.00001))

        if self.L1_norm != 0.0:
            cost += (2 * TT.sum(self.W) + TT.sum(self.b_hidden)
                     + TT.sum(self.b_visible)) * self.L1_norm
        if self.bias_decay != 0.0:
            extra_bias_decay = (TT.sum(self.b_hidden ** 2) + TT.sum(self.b_visible ** 2)) * self.bias_decay
            cost += extra_bias_decay

        #############################
        # The training updates part #
        #############################
        
        check_kwargs(kwargs, ['learning_rate'])

        learning_rate = kwargs['learning_rate']

        # (Too connected with gradient cost/monitoring cost mess-up
        # to keep separate...)
        # --connection through consider_constant=[chain_end].
        #   Couldn't we convince Theano otherwise that chain_end is a constant?
        #   Like, saying so in the cost expression? *Before* the cost 
        #   expression?
        
        gparams = TT.grad(cost, self.params, consider_constant=[chain_end])
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # updates already contains the Theano rng update
            updates[param] = param - gparam * learning_rate       
        
        # We need an extra update of the persistent Markov chains
        # if we're running PCD, plus we're using a different monitoring
        # cost expression for CD vs. PCD.
        if self.persistent:
            # Note that this works only if persistent is a shared variable
            updates[self.persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self._get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD.
            # This is the only place where we actually use the pre-sigmoid
            # activation we are returning all the way from mean_v_given_h.
            # But it still *is* necessary.
            # ...this is the second point of extra coupling between getting 
            # the training cost expression and getting the training updates.
            # Can we get rid of the 'updates' in the call to _get_rec_cost?
            #  ...no, because we'll be again updating the **** theano RNG
            # in the process.
            # ...or maybe not? When we have pre_sigmoid_nvs...
            # How can we pass pre_sigmoid_nvs[-1]?
            monitoring_cost = self._get_reconstruction_cost(X, #updates,
                                                        pre_sigmoid_nvs[-1])
            
        # Keeping track of things
        self._training_cost = cost
        self._precomputed_training_updates = updates
        self._monitoring_cost = monitoring_cost
        self._cost_and_updates_computed = True
        
        return monitoring_cost, updates # Does this need a return value, anyway?
    
    def _cost(self, X):
        """Returns the MONITORING cost expression, which is used by the 
        learner  to monitor progress, validate, check on improvements, 
        etc. 
        
        This behavior is NOT in line with how _cost usually works.
        
        .. warn:
        
            Depends on the :func:`_cost_and_training_updates()` method being
            run before  _cost() is called!
            
        .. note:
        
            This is all a facade to keep the interface of the RBM consistent 
            with the others. The internal workings are a true Theano-inflicted
            giant  mess-up. Will (maybe) deal with it later; for now I gave up
            and just copied over the solution from deeplearning.net tutorials.
        """
        if not self._cost_and_updates_computed:
            raise AttributeError("Before calling _training_updates, you should\
             definitely call  _cost_and_updates_computed()")
            
        return self._monitoring_cost
    
    def _training_updates(self, **kwargs):
        """Returns the updates dictionary necessary to train the RBM. 
        
         .. warn:
        
            Depends on the :func:`_cost_and_training_updates()` method being
            run before  _cost() is called!
            
        .. note:
        
            This is all a facade to keep the interface of the RBM consistent 
            with the others. The internal workings are a true Theano-inflicted
            giant  mess-up. Will (maybe) deal with it later; for now I gave up
            and just copied over the solution from deeplearning.net tutorials.
            
        .. warn:
        
            Don't do anything with these updates except use them in setup() to
            pass to ``theano.function`` as the ``updates`` parameter for the
            RBM training function.
        """
        if not self._cost_and_updates_computed:
            raise AttributeError("Before calling _training_updates, you should\
             definitely call  _cost_and_updates_computed()")
            
        return self._precomputed_training_updates

    def _get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood
        
        Copied over from deeplearning.net. I have little idea about
        what it does, why the ``updates`` are passed here, etc.
        """

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = TT.iround(self.inputs)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = TT.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = TT.mean(self.n_in * TT.log(TT.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_in

        return cost
        
    def _get_reconstruction_cost(self, chain_start_v,
                                  pre_sigmoid_chain_end_v):
        """Computes cross-entropy between visible units' at chain start
        and at chain end. Uses the pre-sigmoid activation Theano
        optimization hack.
        """
        chain_end_v = self.backward_activation(pre_sigmoid_chain_end_v)
        # We're not using the monitoring cost in theano.grad, so we can
        # use the numpy rng and thus get rid of the updates.
        #chain_end_v_sample = self.theano_rng.binomial(size=chain_end_v.shape,
        #                                              n=1, p=chain_end_v,
        #                                       dtype = theano.config.floatX)
        
        return -TT.sum(chain_start_v * TT.log(chain_end_v) +
                (1 - chain_start_v) * TT.log(1 - chain_end_v))
    
                
    def error(self, X):
        return self._cost(X)

    def _sparsity_cross_entropy(self, X):
        """
        Computes the KL divergence of distribution of the sparsity target
        w.r.t. mean activation of each hidden neuron.

        :param X: The input data batch.

        :return: The KL-divergence... (see desc.)
        """
        mean_act = TT.mean(self.activation(TT.dot(X, self.W) + self.b_hidden), axis=0)
        mean_act_compl = 1.0 - mean_act
        rho_term = mean_act * TT.log(mean_act / self.sparsity_target)
        neg_rho_term = mean_act_compl * TT.log(mean_act_compl / (1.0 - self.sparsity_target))
        kl_divergence = TT.sum(rho_term + neg_rho_term)

        return kl_divergence

    def _output_sparsity_cross_entropy(self, X):
        """
        Computes the KL divergence of distribution of the sparsity target
        w.r.t. mean activation of each hidden neuron.

        :param X: The input data batch.

        :return: The KL-divergence... (see desc.)
        """
        mean_act = TT.mean(self.activation(TT.dot(X, self.W) + self.b_hidden), axis=1)
        mean_act_compl = 1.0 - mean_act
        rho_term = mean_act * TT.log(mean_act / self.output_sparsity_target)
        neg_rho_term = mean_act_compl * TT.log(mean_act_compl / (1.0 - self.output_sparsity_target))
        kl_divergence = TT.sum(rho_term + neg_rho_term)

        return kl_divergence

    def _entropy_cost(self, X):
        """Computes the sum of entropies of the bernoulli distributions with
        hidden activations as means. (Divided by size of minibatch.)"""
        #entropy = -1.0 * (X * TT.log(X) + (1.0 - X) * TT.log(1.0 - X))
        entropies = theano.nnet.binary_crossentropy(X, X)
        rowsum = TT.sum(entropies, axis=1)
        return TT.mean(rowsum)

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

        model._cost_and_training_updates(model.inputs,
                                          learning_rate=learning_rate)
                
        bound_cost = model._cost(model.inputs)
        updates = model._precomputed_training_updates(cost = bound_cost,
                                          learning_rate = learning_rate)
        
        # Notice the trick in givens = {}: we link the data to the
        # supervised_model_instance's inputs, so that the data runs
        # through the previous layers first and gets correctly transformed.
        batch_index = TT.lscalar('batch_index')
        pretrain_model = theano.function(inputs = [model.inputs],
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

        :type data: Dataset
        :param data: The dataset on which the model will be run.

        :type model: BaseUnsupervisedModel
        :param model: A model instance that the setup should use.

        :type batch_size: int
        :param batch_size: how many data items will be in one minibatch
        (the data is split to minibatches for training,
        validation and testing)

        :type learning_rate: theano.config.floatX
        :param learning_rate: How fast will the model move in the direction
        of the gradient.

        :type model_init_kwargs: kwargs
        :param model_init_kwargs: Various keyword arguments that get passed
        to the model constructor. See constructor
        argument documentation.
                
        :rtype: ModelHandle                     
        :returns: ``ModelHandle(model, train_f, validate_f, test_f)``
        where ``model`` is the Model instance initialized during
        :func:`setup` and the ``_func`` variables are compiled
        theano.functions to use in a learner.

        """
        logging.info('Setting up RBM model.')

        index = TT.lscalar() # index of minibatch
        X = TT.matrix('X')   # data as a matrix
        # There is no response vector.
       
        # Check for kwargs ... obsolete?
#       cls._check_init_args(model_init_kwargs)
        # Debugging...
        training_kwargs = {}
        if heavy_debug:
            training_kwargs['mode'] = theano.compile.MonitorMode(
                        post_func=safire.utils.merciless_print).excluding(
                                            'local_elemwise_fusion', 'inplace')

        dummy_inputs = data.train_X_batch(0, 2).astype(theano.config.floatX)
        #print dummy_inputs
        X.tag.test_value = dummy_inputs

        #print 'Dummy inputs:', X.tag.test_value

        # Construct the model instance, or use supplied and do sanity checks.
        if model is None:

            if 'n_out' in model_init_kwargs:
                n_out = model_init_kwargs['n_out']
            elif hasattr(data, 'n_out') and isinstance(data.n_out, int):
                model_init_kwargs['n_out'] = data.n_out
            else:
                raise ValueError('Must supply n_out either from dataset or **model_init_kwargs.')

        logging.info('Initializing model.')

        # Construct the model instance, or use supplied and do sanity checks.
        if model is None:
            logging.info('\t(from scratch)')
            model = cls(inputs=X, n_in = data.n_in,
                        **model_init_kwargs)
        else:
            # Sanity (dimensionality...) checks: 
            # - Are we passing a model of the same type as we're trying
            #   to set up?
            # - Are we passing a dataset that the model can work on?
            assert cls == type(model)
            assert model.n_in == data.n_in
            # assert model.n_out == data.n_out
            model.inputs = X

        logging.info('\tPreparing cost and training updates...')

        # Key difference for RBM setup: have to call the **()@!*!#*!!!#$!!
        # _cost_and_training_updates() method
        model._cost_and_training_updates(model.inputs, learning_rate=learning_rate)

        # The 'X' variable has no role on the returned expression here.
        bound_cost = model._cost(model.inputs)

        # The 'cost' parameter is perfectly obsolete here.
        updates = model._training_updates(cost=bound_cost,
                                          learning_rate=learning_rate)

        # Compile a Theano function that trains the model: returns the cost
        # and updates the model parameters based on the rules defined by the
        # model._training_updates() function.
        train_model = theano.function(inputs = [model.inputs],
                                      outputs = bound_cost,
                                      updates = updates,
                                      allow_input_downcast=True,
                                      **training_kwargs)

        # Compile a Theano function that computes the cost that are made
        # by the model on a minibatch of devel/test data
        validate_model = theano.function(inputs = [model.inputs],
                                outputs = model.error(model.inputs),
                                allow_input_downcast=True)
        
        test_model = theano.function(inputs = [model.inputs],
                                outputs = model.error(model.inputs),
                                allow_input_downcast=True)

        run_model = theano.function(inputs = [model.inputs],
                                    outputs = model.outputs,
                                    allow_input_downcast=True)

        return ModelHandle(model, train_model, validate_model, test_model,
                           run_model)

        
        
