"""
This module contains classes that ...
"""
import logging
import numpy
import theano

__author__ = "Jan Hajic jr."


class SomeClass(object):
    """This class ...
    """

    def __init__(self):
        pass


def standard_sgd_updater(params, gradients, learning_rate):

    updates = [ (param, param - learning_rate * gradient)
                for param, gradient in zip (params, gradients) ]
    return updates


class StandardSGDUpdater(object):
    def __init__(self, learning_rate=0.13):
        self.learning_rate = learning_rate

    def __call__(self, params, gradients):
        updates = []
        for param, gradient in zip(params, gradients):
            logging.debug('Updater processing param: %s' % str(param))
            newparam = param - self.learning_rate * gradient
            updates.append((param, newparam))
        return updates


class MomentumUpdater(object):
    """Doesn't currently do anything."""
    def __init__(self, learning_rate=0.13, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def __call__(self, params, gradients):
        updates = []
        for param, gradient in zip(params, gradients):
            logging.debug('Updater processing param: %s' % str(param))
            newparam = param - self.learning_rate * gradient
            updates.append((param, newparam))
        return updates


class ResilientBackpropUpdater(object):

    def __init__(self, params, init_update=0.05, positive_step=1.2,
                 negative_step=0.5, max_step=1.0, min_step=0.0001):
        """Initializes the rprop updater with all the intermediate
        data as Theano shared variables."""
        logging.debug('Initializing rprop updater.')

        self.init_update = init_update
        self.positive_step = positive_step
        self.negative_step = negative_step
        self.max_step = max_step
        self.min_step = min_step

        self.deltas = [] # Will rely on consistent parameter ordering
        self.last_changes = []
        self.last_grads = []

        # Used to "tie down" symbolic gradient updates to shared variables.
        # Never gets updated.
        self._last_grad_shared_bases = []

        # Initialize the update arguments
        for param in params:
            delta = numpy.ones(param.get_value(borrow=True).shape,
                                      dtype=param.dtype) * self.init_update
            shared_delta = theano.shared(delta,
                                         name=param.name + '_rprop_delta')
            self.deltas.append(shared_delta)

            last_change = numpy.zeros(param.get_value(borrow=True).shape,
                                      dtype=param.dtype)
            shared_last_change = theano.shared(last_change,
                                        name=param.name + '_rprop_lastchange')
            self.last_changes.append(shared_last_change)

            last_gradient = numpy.ones(param.get_value(borrow=True).shape,
                                      dtype=param.dtype) * self.init_update
            shared_lastgrad = theano.shared(last_gradient,
                                         name=param.name + '_rprop_lastgrad')
            self.last_grads.append(shared_lastgrad)

            lastgrad_base = numpy.ones(param.get_value(borrow=True).shape,
                                      dtype=param.dtype) * self.init_update
            shared_lastgrad_base = theano.shared(lastgrad_base,
                                         name=param.name + '_rprop_lastgrad_base')
            self._last_grad_shared_bases.append(shared_lastgrad_base)


    def __call__(self, params, gradients, **kwargs):
        """Implements the update formula for a RPROP step."""

        logging.info('Initializing rprop updates.')

        updates = []
        for param, gradient, lastgrad, lastgrad_base, delta, lastchange in zip(
                                                      params,
                                                      gradients,
                                                      self.last_grads,
                                                      self._last_grad_shared_bases,
                                                      self.deltas,
                                                      self.last_changes):
            logging.info('Creating rprop updates for param %s' % param)
            # DUMMY IMPLEMENTATION
            #print 'Gradient: ', dir(gradient)
            #print 'Gradient size:', theano.pprint(gradient.size)
            change = theano.tensor.sgn(gradient * lastgrad)
            #print 'Signum: ', theano.pprint(change)
            if theano.tensor.gt(change, 0): # Needs to be rewritten as theano-IF
                new_delta = theano.tensor.minimum(delta * self.positive_step,
                                              self.max_step)
                weight_change = theano.tensor.sgn(gradient) * new_delta
                lastgrad = gradient * lastgrad_base

            elif theano.tensor.lt(change, 0):
                new_delta = theano.tensor.maximum(delta * self.negative_step,
                                              self.min_step)
                weight_change = -lastchange
                lastgrad = theano.tensor.zeros_like(gradient, dtype=gradient.dtype)
            else:
                new_delta = delta
                weight_change = theano.tensor.sgn(gradient) * delta
                lastgrad = theano.tensor.ones_like(param, dtype=param.dtype)

            updates.append((param, param - weight_change))
            logging.info('Creating delta update for %s of type %s:' % (
                str(delta), str(type(delta))))
            updates.append((delta, new_delta))

            logging.info('Creating lastgrad update for %s of type %s:' % (
                str(lastgrad), str(type(lastgrad))))
            updates.append((lastgrad, gradient))

            logging.info('Creating weights change update for %s of type %s:' % (
                str(lastchange), str(type(lastchange))))
            updates.append((lastchange, weight_change))

            #lastchange = weight_change ?? why was this here?

        logging.info('RpropUpdater: updates complete.')

        return updates