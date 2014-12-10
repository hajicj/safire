"""
.. module:: 
    :platform: Unix
    :synopsis: ???

.. moduleauthor: Jan Hajic <hajicj@gmail.com>
"""

import numpy

import theano
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams
from safire.learning.models import RestrictedBoltzmannMachine

from safire.learning.models.base_unsupervised_model import BaseUnsupervisedModel
from safire.learning.interfaces.pretraining_model_handle import PretrainingModelHandle
from safire.learning.interfaces import ModelHandle
from safire.utils import check_kwargs

class ReplicatedSoftmax(RestrictedBoltzmannMachine):
    def __init__(self, inputs, n_in, n_out=100,
                 activation=TT.nnet.sigmoid,
                 backward_activation=TT.nnet.softmax,
                 W=None, b=None, b_hidden=None, b_visible=None,
                 persistent=None, CD_k=1, CD_use_mean=True,
                 sparsity_target=None, output_sparsity_target=None,
                 numpy_rng=numpy.random.RandomState(),
                 L1_norm=0.0, L2_norm=0.0, bias_decay=0.0,
                 entropy_loss=0.0, centering=False, prefer_extremes=False,
                 theano_rng=None):
        """ Initialize the parameters of the Replicated Softmax. This merely
        sets the correct values for an RBM in the defaults. (The only other
        difference than using the specific pair of forward/backward activations
        is the computation of free energy.)

        .. note::

          In order for this model to implement the real Replicated Softmax
          Model of Salakhtudinov and Hinton, the ``activation`` and
          ``backward_activation`` parameters have to remain in their default
          form.

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
        :param activation: The nonlinearity applied at neuron
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
        """
        super(ReplicatedSoftmax, self).__init__(
            inputs,
            n_in,
            n_out,
            TT.nnet.sigmoid,
            TT.nnet.softmax,
            W, b, b_hidden, b_visible,
            persistent, CD_k, CD_use_mean, sparsity_target,
            output_sparsity_target,
            numpy_rng, L1_norm, L2_norm, bias_decay,
            entropy_loss, centering, prefer_extremes,
            theano_rng)

        print 'B: ', self.b_hidden.broadcastable
        self.b_hidden_broadcastable = TT.row('b_hidden_broadcastable',
                                             dtype=theano.config.floatX)
        print 'B/dsh: ', self.b_hidden.broadcastable
        print 'Hidden type:', type(self.b_hidden)
        TT.addbroadcast(self.b_hidden_broadcastable, 0)
        self.b_hidden_broadcastable.tag.test_value = numpy.ones((2, self.n_out))
        print 'B/dsh: ', self.b_hidden.broadcastable
        # Need this for broadcasting the bias
                                          # vector to multiply by document sizes
                                          # in free_energy.

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
        print 'Running free energy.'

        D = TT.sum(visible, axis=1)
        exponent_term = TT.dot(visible, self.W) + TT.outer(D, self.b_hidden)
                        # TT.outer(D, self.b_hidden)
                        # D is a coefficient, b_hidden should

        hidden_term = TT.sum(TT.log(1 + TT.exp(exponent_term)), axis=1)

        # This is the other and more crucial difference between an RBM and a
        #  RSM: multiplying hidedn bias by "document length".
        b_visible_term = TT.dot(visible, self.b_visible)

        free_energy = - hidden_term - b_visible_term
        return free_energy
